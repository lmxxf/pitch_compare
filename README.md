# Pitch Compare - 音准对比分析工具

将原唱和学生演唱的音频转换为可视化图表，方便用AI（如Gemini）分析音准和节奏问题。

## 为什么用图表而不是音频？

AI的视觉模态分辨率比音频模态高。直接给Gemini看音高曲线对比图，比让它"听"mp3能获得更精确的分析。

## 功能

- **人声分离**：使用demucs从伴奏中提取纯人声
- **音高提取**：使用librosa.pyin提取音高曲线
- **DTW对齐**：自动对齐不同长度的音频（处理节奏差异）
- **可视化输出**：
  - 音高曲线对比图（原唱 vs 学生）
  - 频谱图对比
  - 节奏分析图
  - 数值分析报告

## 安装

```bash
pip install -r requirements.txt
```

需要中文字体支持：
```bash
# Ubuntu/Debian
sudo apt install fonts-wqy-microhei

# 清除matplotlib字体缓存
rm -rf ~/.cache/matplotlib
```

## 使用

```bash
python pitch_compare.py 原唱.mp3 学生.mp3 -o output/
```

### 参数

- `原唱.mp3`：原唱音频文件
- `学生.mp3`：学生演唱的音频文件
- `-o, --output`：输出目录（默认：output）
- `--no-separation`：跳过人声分离（如果输入已经是纯人声）

### 输出文件

- `pitch_comparison.png`：音高对比图（最重要）
  - 原始音高曲线（未对齐）
  - DTW对齐后的音高曲线
  - 音高偏差分析（绿=准，橙=略偏，红=严重偏）
- `spectrogram_comparison.png`：频谱图对比
- `rhythm_analysis.png`：节奏分析（偏离对角线=快了/慢了）
- `analysis.txt`：数值报告

## 示例输出

### 音高对比图

![Pitch Comparison](examples/pitch_comparison.png)

- 上图：原始音高曲线（蓝=原唱，红=学生）
- 中图：DTW对齐后的音高曲线
- 下图：音高偏差（绿=准，橙=略偏，红=严重偏）

### 频谱图对比

![Spectrogram Comparison](examples/spectrogram_comparison.png)

### 节奏分析

![Rhythm Analysis](examples/rhythm_analysis.png)

偏离对角线表示节奏偏差（快了或慢了）

### 分析报告

```
音准分析报告
================

整体统计：
- 平均偏差: -222.4 cents （偏低）
- 偏差标准差: 385.7 cents
- 中位数偏差: -170.0 cents

音准分布：
- 准确 (<25 cents): 268 帧 (7.4%)
- 略偏 (25-50 cents): 187 帧 (5.1%)
- 严重偏 (>50 cents): 3185 帧 (87.5%)

总体评价：
需要大量练习。仅7%的音准在可接受范围内。
整体趋势：偏低。建议增加气息支撑，提高音调。
```

## 如何解读

### 音高偏差（cents）

- **< 25 cents**：准确，一般听众难以察觉
- **25-50 cents**：略偏，1/4到1/2半音
- **> 50 cents**：严重偏，明显跑调
- **100 cents = 1个半音**

### 给AI分析

将输出的4个文件发送给Gemini，它能看到：
- 哪个时间点音准偏了
- 偏高还是偏低
- 节奏是快了还是慢了
- 整体水平评估

## 技术栈

- **demucs**：Meta的人声分离模型
- **librosa**：音频分析库
- **matplotlib**：可视化

## License

MIT
