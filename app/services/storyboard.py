"""
Storyboard 来源的视频生成：图片完全由用户从素材库手动挑选，不做 AI 生图。

流程：逐分镜台词单独跑 TTS 拿到真实音频时长 -> 按该时长把分镜图片转成视频
片段 -> ffmpeg 无损拼接所有片段 -> ffmpeg 拼接所有分镜音频 -> 按累计偏移量
手写 SRT 字幕 -> 交给 video.generate_video 完成最终字幕烧录/配乐/编码。

不复用 video.combine_videos，因为它会按 max_clip_duration 裁剪或循环素材，
会破坏这里"每张图恰好显示它所在分镜的真实配音时长"的时间轴。

音频拼接直接用 ffmpeg 的 concat demuxer，不用 pydub：pydub 的 from_file()
默认会先跑一次 ffprobe 探测元数据（mediainfo_json），很多环境（包括这里）
只打包了 ffmpeg 没有 ffprobe，会在拼接第一步就以 WinError 2 失败。
"""

import os
import subprocess
from typing import Any, List

from loguru import logger

from app.models.schema import StoryboardScene, VideoParams
from app.services import material, video, voice
from app.utils import utils


def synthesize_scene_audio(
    task_id: str, params: VideoParams, scenes: List[StoryboardScene]
) -> tuple[List[dict[str, Any]], List[str]]:
    """对每个分镜台词单独合成语音，返回 (scene_infos, audio_files)。

    scene_infos 里每项包含 scene、dialogue、duration（秒）、start（累计偏移秒）。
    任一分镜合成失败就整体返回空列表，避免用不完整的时间轴继续往下走。
    """
    task_dir = utils.task_dir(task_id)
    scene_infos: List[dict[str, Any]] = []
    audio_files: List[str] = []
    offset = 0.0

    for index, scene in enumerate(scenes):
        dialogue = (scene.dialogue or "").strip()
        if not dialogue:
            continue

        voice_file = os.path.join(task_dir, f"storyboard-voice-{index}.mp3")
        try:
            sub_maker = voice.tts(
                text=dialogue,
                voice_name=params.voice_name,
                voice_rate=params.voice_rate,
                voice_file=voice_file,
                voice_volume=params.voice_volume,
            )
        except Exception as e:
            logger.error(
                f"failed to synthesize storyboard scene voice: index={index}, error={str(e)}"
            )
            return [], []

        if not sub_maker or not (
            os.path.exists(voice_file) and os.path.getsize(voice_file) > 0
        ):
            logger.error(f"storyboard scene voice synthesis returned no audio: index={index}")
            return [], []

        duration = voice.get_audio_duration(sub_maker)
        if not duration:
            duration = voice.get_audio_duration(voice_file)
        if not duration:
            logger.error(f"could not determine storyboard scene audio duration: index={index}")
            return [], []

        scene_infos.append(
            {
                "index": index,
                "scene": scene,
                "dialogue": dialogue,
                "duration": duration,
                "start": offset,
            }
        )
        audio_files.append(voice_file)
        offset += duration

    return scene_infos, audio_files


def concat_audio_files(audio_files: List[str], output_path: str) -> bool:
    """用 ffmpeg concat demuxer 拼接分镜配音，按顺序无缝首尾相接。"""
    if not audio_files:
        return False

    output_dir = os.path.dirname(output_path) or "."
    concat_list_file = os.path.join(output_dir, "ffmpeg-audio-concat-list.txt")
    try:
        with open(concat_list_file, "w", encoding="utf-8") as fp:
            for audio_file in audio_files:
                absolute_path = os.path.abspath(audio_file).replace("\\", "/")
                escaped_path = absolute_path.replace("'", "'\\''")
                fp.write(f"file '{escaped_path}'\n")

        command = [
            utils.get_ffmpeg_binary(),
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_file,
            # 不同分镜的配音可能来自不同 TTS 请求，逐段重新编码成统一参数，
            # 避免 concat demuxer 在流复制模式下要求所有输入编码参数完全一致。
            "-c:a",
            "libmp3lame",
            output_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_message = (result.stderr or result.stdout or "").strip()
            logger.error(f"failed to concat storyboard narration audio: {error_message}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        logger.error(f"failed to concat storyboard narration audio: {str(e)}")
        return False
    finally:
        try:
            os.remove(concat_list_file)
        except Exception:
            pass


def build_subtitle_file(scene_infos: List[dict[str, Any]], subtitle_path: str) -> str:
    lines = []
    for idx, info in enumerate(scene_infos, start=1):
        lines.append(
            utils.text_to_srt(
                idx,
                info["dialogue"],
                info["start"],
                info["start"] + info["duration"],
            )
        )
    try:
        with open(subtitle_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return subtitle_path
    except Exception as e:
        logger.error(f"failed to write storyboard subtitle file: {str(e)}")
        return ""


def build_scene_video_clips(
    task_id: str, scene_infos: List[dict[str, Any]], params: VideoParams
) -> List[str]:
    """把每个分镜选中的库内图片按该分镜的配音时长转换成视频片段，按顺序返回。

    一个分镜可以选多张图，此时该分镜时长会在这些图片间平均分配。所有片段都
    会缩放到同一个 video_aspect 分辨率，否则 ffmpeg 拼接不同尺寸的片段会失败。
    """
    task_dir = utils.task_dir(task_id)
    clip_paths: List[str] = []

    for info in scene_infos:
        scene: StoryboardScene = info["scene"]
        images = scene.images or []
        if not images:
            logger.error(f"storyboard scene has no selected image: index={info['index']}")
            return []

        per_image_duration = max(info["duration"] / len(images), 0.1)
        for image_filename in images:
            try:
                clip_path = material.scene_image_to_video(
                    image_filename=image_filename,
                    save_dir=task_dir,
                    duration=per_image_duration,
                    video_aspect=params.video_aspect,
                )
            except Exception as e:
                logger.error(
                    "failed to build storyboard scene video clip: "
                    f"index={info['index']}, image={image_filename}, error={str(e)}"
                )
                return []
            if not clip_path:
                logger.error(
                    "failed to build storyboard scene video clip: "
                    f"index={info['index']}, image={image_filename}"
                )
                return []
            clip_paths.append(clip_path)

    return clip_paths


def combine_storyboard_clips(
    task_id: str, clip_paths: List[str], threads: int, output_path: str
) -> bool:
    task_dir = utils.task_dir(task_id)
    try:
        video.concat_video_clips_with_ffmpeg(
            clip_files=clip_paths,
            output_file=output_path,
            threads=threads or 2,
            output_dir=task_dir,
        )
    except Exception as e:
        logger.error(f"failed to concat storyboard video clips: {str(e)}")
        return False
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0
