#!/usr/bin/env python3
"""
Pitch Compare - 音准对比分析工具
输入：原唱mp3 + 学生mp3
输出：音高对比图、频谱图、节奏分析、数值报告
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
import soundfile as sf
from scipy.spatial.distance import cdist

# 中文字体支持
def setup_chinese_font():
    """设置中文字体，优先使用系统字体，否则用英文"""
    import subprocess

    # 尝试常见中文字体
    chinese_fonts = [
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'Noto Sans CJK SC',
        'Noto Sans SC',
        'SimHei',
        'Microsoft YaHei',
        'PingFang SC',
        'Hiragino Sans GB',
    ]

    # 获取系统可用字体
    available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)

    for font in chinese_fonts:
        if font in available_fonts:
            matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"使用中文字体: {font}")
            return True

    # 没找到中文字体，用英文
    print("警告: 未找到中文字体，图表将使用英文")
    return False

CHINESE_FONT_AVAILABLE = setup_chinese_font()


def separate_vocals(audio_path: str, output_dir: str) -> str:
    """使用demucs分离人声"""
    print(f"分离人声: {audio_path}")

    # 运行demucs
    cmd = [
        "demucs",
        "--two-stems", "vocals",  # 只分离人声
        "-o", output_dir,
        audio_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"demucs stderr: {result.stderr}")
        print(f"demucs stdout: {result.stdout}")
        raise RuntimeError(f"demucs failed with code {result.returncode}")

    # 找到输出的vocals文件
    audio_name = Path(audio_path).stem
    vocals_path = os.path.join(output_dir, "htdemucs", audio_name, "vocals.wav")

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"人声分离失败: {vocals_path}")

    return vocals_path


def extract_pitch(audio_path: str, sr: int = 22050) -> tuple:
    """提取音高曲线，使用librosa.pyin"""
    print(f"提取音高: {audio_path}")

    y, sr = librosa.load(audio_path, sr=sr)

    # 使用pyin提取音高
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sr
    )

    times = librosa.times_like(f0, sr=sr)

    return times, f0, y, sr


def hz_to_cents(f0: np.ndarray, ref_hz: float = 440.0) -> np.ndarray:
    """将Hz转换为cents（相对于参考音高）"""
    with np.errstate(divide='ignore', invalid='ignore'):
        cents = 1200 * np.log2(f0 / ref_hz)
    return cents


def hz_to_note_name(hz: float) -> str:
    """将Hz转换为音符名"""
    if np.isnan(hz) or hz <= 0:
        return ""
    note_num = 12 * np.log2(hz / 440.0) + 69
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_idx = int(round(note_num)) % 12
    octave = int(round(note_num)) // 12 - 1
    return f"{note_names[note_idx]}{octave}"


def align_with_dtw(times1, f0_1, times2, f0_2):
    """使用DTW对齐两个音高序列"""
    print("DTW对齐中...")

    # 转换为cents进行对比（忽略nan）
    cents1 = hz_to_cents(f0_1)
    cents2 = hz_to_cents(f0_2)

    # 用0填充nan（DTW计算用）
    cents1_filled = np.nan_to_num(cents1, nan=0)
    cents2_filled = np.nan_to_num(cents2, nan=0)

    # 计算DTW
    D, wp = librosa.sequence.dtw(
        cents1_filled.reshape(1, -1),
        cents2_filled.reshape(1, -1),
        metric='euclidean'
    )

    # wp是对齐路径，shape=(N, 2)
    # wp[:, 0]是原唱索引，wp[:, 1]是学生索引

    return wp, D


def calculate_pitch_diff(f0_1, f0_2, wp):
    """计算对齐后的音高差异"""
    diffs = []

    for i, j in wp:
        if not np.isnan(f0_1[i]) and not np.isnan(f0_2[j]):
            # cents差异
            diff_cents = 1200 * np.log2(f0_2[j] / f0_1[i])
            diffs.append(diff_cents)
        else:
            diffs.append(np.nan)

    return np.array(diffs)


def plot_pitch_comparison(times1, f0_1, times2, f0_2, wp, output_path):
    """绘制音高对比图"""
    print("绘制音高对比图...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 图1：原始音高曲线
    ax1 = axes[0]
    ax1.plot(times1, f0_1, 'b-', label='原唱', alpha=0.7, linewidth=1)
    ax1.plot(times2, f0_2, 'r-', label='学生', alpha=0.7, linewidth=1)
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('频率 (Hz)')
    ax1.set_title('原始音高曲线（未对齐）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 图2：DTW对齐后的音高对比
    ax2 = axes[1]

    # 根据DTW路径重新采样
    aligned_times = times1[wp[:, 0]]
    aligned_f0_1 = f0_1[wp[:, 0]]
    aligned_f0_2 = f0_2[wp[:, 1]]

    ax2.plot(aligned_times, aligned_f0_1, 'b-', label='原唱', alpha=0.7, linewidth=1)
    ax2.plot(aligned_times, aligned_f0_2, 'r-', label='学生', alpha=0.7, linewidth=1)
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('频率 (Hz)')
    ax2.set_title('DTW对齐后的音高曲线')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 图3：音高差异（cents）
    ax3 = axes[2]
    diffs = calculate_pitch_diff(f0_1, f0_2, wp)

    # 颜色编码：绿色=准，黄色=略偏，红色=严重偏
    colors = []
    for d in diffs:
        if np.isnan(d):
            colors.append('gray')
        elif abs(d) < 25:  # <25 cents = 准
            colors.append('green')
        elif abs(d) < 50:  # 25-50 cents = 略偏
            colors.append('orange')
        else:  # >50 cents = 严重偏
            colors.append('red')

    ax3.scatter(aligned_times, diffs, c=colors, s=1, alpha=0.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(y=50, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.axhline(y=-50, color='orange', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.axhline(y=100, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.axhline(y=-100, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('时间 (秒)')
    ax3.set_ylabel('音高偏差 (cents)')
    ax3.set_title('音高偏差分析（正=偏高，负=偏低）')
    ax3.set_ylim(-200, 200)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return diffs


def plot_spectrogram_comparison(y1, sr1, y2, sr2, output_path):
    """绘制频谱图对比"""
    print("绘制频谱图...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 原唱频谱
    D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
    librosa.display.specshow(D1, sr=sr1, x_axis='time', y_axis='hz', ax=axes[0])
    axes[0].set_title('原唱频谱图')
    axes[0].set_ylim(0, 2000)  # 人声主要频率范围

    # 学生频谱
    D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
    librosa.display.specshow(D2, sr=sr2, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('学生频谱图')
    axes[1].set_ylim(0, 2000)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rhythm_analysis(times1, times2, wp, output_path):
    """绘制节奏分析图"""
    print("绘制节奏分析图...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # DTW路径显示节奏差异
    path_times1 = times1[wp[:, 0]]
    path_times2 = times2[wp[:, 1]]

    # 理想情况是对角线
    ax.plot(path_times1, path_times2, 'b-', linewidth=0.5, alpha=0.7)

    # 对角线参考
    max_time = max(times1[-1], times2[-1])
    ax.plot([0, max_time], [0, max_time], 'k--', linewidth=1, alpha=0.5, label='理想节奏')

    ax.set_xlabel('原唱时间 (秒)')
    ax.set_ylabel('学生时间 (秒)')
    ax.set_title('节奏对齐分析（偏离对角线=节奏偏差）')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(f0_1, f0_2, wp, diffs, output_path):
    """生成分析报告"""
    print("生成分析报告...")

    # 过滤nan
    valid_diffs = diffs[~np.isnan(diffs)]

    if len(valid_diffs) == 0:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("无法生成报告：没有有效的音高数据\n")
        return

    # 统计
    mean_diff = np.mean(valid_diffs)
    std_diff = np.std(valid_diffs)
    median_diff = np.median(valid_diffs)

    # 分类统计
    accurate = np.sum(np.abs(valid_diffs) < 25)  # <25 cents
    slightly_off = np.sum((np.abs(valid_diffs) >= 25) & (np.abs(valid_diffs) < 50))
    seriously_off = np.sum(np.abs(valid_diffs) >= 50)

    total = len(valid_diffs)

    report = f"""音准分析报告
================

整体统计：
- 平均偏差: {mean_diff:.1f} cents （{'偏高' if mean_diff > 0 else '偏低'}）
- 偏差标准差: {std_diff:.1f} cents
- 中位数偏差: {median_diff:.1f} cents

音准分布：
- 准确 (<25 cents): {accurate} 帧 ({100*accurate/total:.1f}%)
- 略偏 (25-50 cents): {slightly_off} 帧 ({100*slightly_off/total:.1f}%)
- 严重偏 (>50 cents): {seriously_off} 帧 ({100*seriously_off/total:.1f}%)

参考标准：
- 25 cents ≈ 1/4半音，一般听众难以察觉
- 50 cents = 1/2半音，明显偏差
- 100 cents = 1个半音，严重跑调

总体评价：
"""

    # 评价
    accuracy_rate = 100 * accurate / total
    if accuracy_rate > 80:
        report += f"优秀！{accuracy_rate:.0f}%的音准在可接受范围内。\n"
    elif accuracy_rate > 60:
        report += f"良好。{accuracy_rate:.0f}%的音准在可接受范围内，但仍有提升空间。\n"
    elif accuracy_rate > 40:
        report += f"需要练习。只有{accuracy_rate:.0f}%的音准在可接受范围内。\n"
    else:
        report += f"需要大量练习。仅{accuracy_rate:.0f}%的音准在可接受范围内。\n"

    # 整体趋势
    if abs(mean_diff) > 30:
        if mean_diff > 0:
            report += "整体趋势：偏高。建议放松喉咙，降低音调。\n"
        else:
            report += "整体趋势：偏低。建议增加气息支撑，提高音调。\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)


def main():
    parser = argparse.ArgumentParser(description='音准对比分析工具')
    parser.add_argument('original', help='原唱音频文件')
    parser.add_argument('student', help='学生音频文件')
    parser.add_argument('--output', '-o', default='output', help='输出目录')
    parser.add_argument('--no-separation', action='store_true', help='跳过人声分离（如果输入已经是纯人声）')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 临时目录用于demucs输出
    with tempfile.TemporaryDirectory() as temp_dir:
        # 人声分离
        if args.no_separation:
            vocals1 = args.original
            vocals2 = args.student
        else:
            vocals1 = separate_vocals(args.original, temp_dir)
            vocals2 = separate_vocals(args.student, temp_dir)

        # 提取音高
        times1, f0_1, y1, sr1 = extract_pitch(vocals1)
        times2, f0_2, y2, sr2 = extract_pitch(vocals2)

        # DTW对齐
        wp, D = align_with_dtw(times1, f0_1, times2, f0_2)

        # 绘图
        diffs = plot_pitch_comparison(
            times1, f0_1, times2, f0_2, wp,
            os.path.join(args.output, 'pitch_comparison.png')
        )

        plot_spectrogram_comparison(
            y1, sr1, y2, sr2,
            os.path.join(args.output, 'spectrogram_comparison.png')
        )

        plot_rhythm_analysis(
            times1, times2, wp,
            os.path.join(args.output, 'rhythm_analysis.png')
        )

        # 生成报告
        generate_report(
            f0_1, f0_2, wp, diffs,
            os.path.join(args.output, 'analysis.txt')
        )

    print(f"\n分析完成！结果保存在: {args.output}/")
    print("- pitch_comparison.png: 音高对比图")
    print("- spectrogram_comparison.png: 频谱图对比")
    print("- rhythm_analysis.png: 节奏分析图")
    print("- analysis.txt: 分析报告")


if __name__ == '__main__':
    main()
