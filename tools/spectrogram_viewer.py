import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import librosa
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_audio(file_path, sr=22050):
    """加载音频文件"""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    except Exception as e:
        print(f"加载音频文件失败 {file_path}: {e}")
        return None, None

def extract_formants(signal, sr, n_formants=3):
    """提取共振峰频率"""
    # 使用短时傅里叶变换
    stft = librosa.stft(signal, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    
    # 计算频率轴
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # 对每个时间帧找到峰值
    formants = []
    for i in range(magnitude.shape[1]):
        frame = magnitude[:, i]
        # 找到峰值
        peaks, _ = find_peaks(frame, height=np.max(frame)*0.1, distance=50)
        # 选择前n_formants个最强的峰值
        if len(peaks) > 0:
            peak_magnitudes = frame[peaks]
            top_peaks = peaks[np.argsort(peak_magnitudes)[-n_formants:]]
            formant_freqs = freqs[top_peaks]
            formants.append(formant_freqs)
        else:
            formants.append(np.array([]))
    
    return formants, freqs

def create_spectrogram_plot(audio_path, title, time_range=None, save_path=None):
    """创建频谱图"""
    # 加载音频
    y, sr = load_audio(audio_path)
    if y is None:
        return None
    
    # 如果指定了时间范围，截取对应片段
    if time_range:
        start_time, end_time = time_range
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y = y[start_sample:end_sample]
    
    # 计算频谱图
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # 计算时间轴和频率轴
    times = librosa.frames_to_time(np.arange(db.shape[1]), sr=sr, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    # 提取共振峰
    formants, _ = extract_formants(y, sr)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制频谱图
    im = ax.imshow(db, aspect='auto', origin='lower', 
                   extent=[times[0], times[-1], freqs[0], freqs[-1]],
                   cmap='viridis')
    
    # 绘制共振峰
    if formants:
        for i, formant_frame in enumerate(formants):
            if len(formant_frame) > 0:
                time_val = times[i] if i < len(times) else times[-1]
                for j, freq in enumerate(formant_frame):
                    if j < 3:  # 只显示前3个共振峰
                        ax.plot(time_val, freq, 'go', markersize=3, alpha=0.7)
    
    # 连接共振峰形成轨迹
    if formants:
        for j in range(3):  # 前3个共振峰
            formant_trajectory = []
            formant_times = []
            for i, formant_frame in enumerate(formants):
                if len(formant_frame) > j:
                    formant_trajectory.append(formant_frame[j])
                    formant_times.append(times[i] if i < len(times) else times[-1])
            
            if len(formant_trajectory) > 1:
                ax.plot(formant_times, formant_trajectory, 'g-', linewidth=1.5, alpha=0.8)
    
    # 设置标签和标题
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('频率 (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 设置频率轴为kHz
    ax.set_ylim(0, 4000)
    yticks = ax.get_yticks()
    ax.set_yticklabels([f'{y/1000:.1f}' for y in yticks])
    ax.set_ylabel('频率 (kHz)', fontsize=12)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('幅度 (dB)', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"频谱图已保存到: {save_path}")
    
    return fig

def create_comparison_plot(audio_files, titles, time_ranges=None, save_path=None):
    """创建对比频谱图"""
    n_files = len(audio_files)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (audio_path, title) in enumerate(zip(audio_files, titles)):
        if i >= 4:  # 最多显示4个
            break
            
        # 加载音频
        y, sr = load_audio(audio_path)
        if y is None:
            continue
        
        # 如果指定了时间范围，截取对应片段
        if time_ranges and i < len(time_ranges):
            start_time, end_time = time_ranges[i]
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            y = y[start_sample:end_sample]
        
        # 计算频谱图
        stft = librosa.stft(y, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        db = librosa.amplitude_to_db(magnitude, ref=np.max)
        
        # 计算时间轴和频率轴
        times = librosa.frames_to_time(np.arange(db.shape[1]), sr=sr, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # 提取共振峰
        formants, _ = extract_formants(y, sr)
        
        # 绘制频谱图
        im = axes[i].imshow(db, aspect='auto', origin='lower', 
                           extent=[times[0], times[-1], freqs[0], freqs[-1]],
                           cmap='viridis')
        
        # 绘制共振峰轨迹
        if formants:
            for j in range(3):  # 前3个共振峰
                formant_trajectory = []
                formant_times = []
                for k, formant_frame in enumerate(formants):
                    if len(formant_frame) > j:
                        formant_trajectory.append(formant_frame[j])
                        formant_times.append(times[k] if k < len(times) else times[-1])
                
                if len(formant_trajectory) > 1:
                    axes[i].plot(formant_times, formant_trajectory, 'g-', linewidth=2, alpha=0.8)
        
        # 设置标签和标题
        axes[i].set_xlabel('时间 (秒)', fontsize=10)
        axes[i].set_ylabel('频率 (Hz)', fontsize=10)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        
        # 设置频率轴为kHz
        axes[i].set_ylim(0, 4000)
        yticks = axes[i].get_yticks()
        axes[i].set_yticklabels([f'{y/1000:.1f}' for y in yticks])
        axes[i].set_ylabel('频率 (kHz)', fontsize=10)
    
    # 隐藏多余的子图
    for i in range(len(audio_files), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比频谱图已保存到: {save_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="生成语音频谱图")
    parser.add_argument("audio_files", nargs="+", help="音频文件路径")
    parser.add_argument("--titles", nargs="+", help="图片标题")
    parser.add_argument("--time-ranges", nargs="+", help="时间范围 (格式: start,end)")
    parser.add_argument("--output", "-o", help="输出图片路径")
    parser.add_argument("--comparison", action="store_true", help="创建对比图")
    parser.add_argument("--show", action="store_true", help="显示图片")
    
    args = parser.parse_args()
    
    # 处理时间范围
    time_ranges = None
    if args.time_ranges:
        time_ranges = []
        for tr in args.time_ranges:
            start, end = map(float, tr.split(','))
            time_ranges.append((start, end))
    
    # 设置标题
    titles = args.titles if args.titles else [f"音频 {i+1}" for i in range(len(args.audio_files))]
    
    if args.comparison:
        # 创建对比图
        fig = create_comparison_plot(args.audio_files, titles, time_ranges, args.output)
    else:
        # 创建单个频谱图
        for i, audio_file in enumerate(args.audio_files):
            title = titles[i] if i < len(titles) else f"音频 {i+1}"
            time_range = time_ranges[i] if time_ranges and i < len(time_ranges) else None
            output_path = args.output if not args.comparison else None
            
            fig = create_spectrogram_plot(audio_file, title, time_range, output_path)
    
    if args.show:
        plt.show()

if __name__ == "__main__":
    main()

