#! /usr/bin/env python3
# coding: utf-8
# Copyright (c) 2023 oatsu
"""
1. UTAUプラグインのテキストファイルを読み取る。
2. LABファイル→WAVファイル
"""
import logging
import sys
import time
import warnings
import pyworld
from datetime import datetime
from glob import glob
from os import chdir, listdir, makedirs, rename, startfile
from os.path import abspath, basename, dirname, exists, join, relpath, splitext,isfile
from shutil import move,copy
from tempfile import TemporaryDirectory, mkdtemp
from typing import Iterable,List, Union

import colored_traceback.always  # pylint: disable=unused-import
import numpy as np
import utaupy
import yaml
from nnmnkwii.io import hts
from scipy.io import wavfile
from scipy import interpolate
from omegaconf import DictConfig, OmegaConf

# scikit-learn で警告が出るのを無視
warnings.simplefilter('ignore')

# my_package.my_moduleのみに絞ってsys.stderrにlogを出す
logging.basicConfig(stream=sys.stdout,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO)
logging.getLogger('simple_enunu')


# autopep8: off ----------------------------------

# ENUNUのフォルダ直下にあるフォルダやファイルをimportできるようにする
sys.path.append(dirname(__file__))
import enulib

try:
    import torch
except ModuleNotFoundError:
    print('----------------------------------------------------------')
    print('初回起動ですね。')
    print('PC環境に合わせてPyTorchを自動インストールします。')
    print('インストール完了までしばらくお待ちください。')
    print('----------------------------------------------------------')
#    enulib.install_torch.ltt_install_torch(sys.executable)
    enulib.install_torch.pip_install_torch(abspath(sys.executable))
    print('----------------------------------------------------------')
    print('インストール成功しました。')
    print('----------------------------------------------------------\n')
    import torch

# NNSVSをimportできるようにする
if exists(join(dirname(__file__), 'nnsvs-master')):
    sys.path.append(join(dirname(__file__), 'nnsvs-master'))
elif exists(join(dirname(__file__), 'nnsvs')):
    sys.path.append(join(dirname(__file__), 'nnsvs'))
else:
    logging.error('NNSVS directory is not found.')

import nnsvs  # pylint: disable=import-error
from nnsvs.svs import SPSVS
from nnsvs.pitch import lowpass_filter
from nnsvs.gen import gen_spsvs_static_features,gen_world_params,split_streams,get_static_stream_sizes
from utils import enunu2nnsvs  # pylint: disable=import-error

logging.debug('Imported NNSVS module: %s', nnsvs)

# autopep8: on -------------------------------------


def get_project_path(path_utauplugin):
    """
    キャッシュパスとプロジェクトパスを取得する。
    """
    plugin = utaupy.utauplugin.load(path_utauplugin)
    setting = plugin.setting
    # ustのパス
    path_ust = setting.get('Project')
    # 音源フォルダ
    voice_dir = setting['VoiceDir']
    # 音声キャッシュのフォルダ(LABとJSONを設置する)
    cache_dir = setting['CacheDir']

    return path_ust, voice_dir, cache_dir


def estimate_bit_depth(wav: np.ndarray) -> str:
    """
    wavformのビット深度を判定する。
    16bitか32bit
    16bitの最大値: 32767
    32bitの最大値: 2147483647
    """
    # 音量の最大値を取得
    max_gain = np.nanmax(np.abs(wav))
    # 学習データのビット深度を推定(8388608=2^24)
    if max_gain > 8388608:
        return 'int32'
    if max_gain > 8:
        return 'int16'
    return 'float'

def wrapped_enunu2nnsvs(voice_dir, out_dir,config):
    """ENUNU用のディレクトリ構造のモデルをNNSVS用に再構築する。
    """
    # torch.save() の出力パスに日本語が含まれているとセーブできないので、一時フォルダを作ってそこに保存してから移動する。
    with TemporaryDirectory(prefix='.temp-enunu2nnsvs-', dir='.') as temp_dir:
        enunu2nnsvs.main(voice_dir, relpath(temp_dir))
        for path in listdir(temp_dir):
            move(join(temp_dir, path), join(out_dir, path))
    copy(
        join(out_dir, 'kana2phonemes.table'),
        basename(config.table_path))
    
    return config


def packed_model_exists(voice_dir: str) -> bool:
    """フォルダ内にNNSVSモデルがあるかどうかを返す

    Args:
        dir (str): Path of the directory
    """
    # SPSVSクラスを使う際に必要なNNSVSモデル用のファイル(の一部)
    required_files = {'config.yaml', 'qst.hed'}
    # 全ての要求ファイルがフォルダ内に存在するか調べて返す
    return all(map(exists, [join(voice_dir, p) for p in required_files]))


def find_table(model_dir: str) -> str:
    """ 歌詞→音素の変換テーブルを探す
    """
    table_files = glob(join(model_dir, '*.table'))
    if len(table_files) == 0:
        raise FileNotFoundError(f'Table file does not exist in {model_dir}.')
    elif len(table_files) > 1:
        logging.warning('Multiple table files are found. : %s', table_files)
    logging.info('Using %s', basename(table_files[0]))
    return table_files[0]

def adjust_wav_gain_for_float32(wav: np.ndarray):
    """
    wavformのビット深度を判定して、float32で適切な音量で出力する。
    16bitか32bit
    16bitの最大値: 32767
    32bitの最大値: 2147483647
    ビット深度を指定してファイル出力(32bit float)

    """
    # 音量の最大値を取得
    max_gain = np.nanmax(np.abs(wav))

    # 学習データのビット深度を推定(8388608=2^24)
    # int32 -> float
    if max_gain > 8388608:
        return wav / 2147483647
    # int16 -> float
    elif max_gain > 8:
        return wav / 32767
    # float
    else:
        return wav
    


def get_standard_function_config(config, key) -> Union[None, str]:
    if 'extensions' not in config:
        return 'built-in'
    return config.extensions.get(key)

def get_extension_path_list(config, key) -> Union[None, List[str]]:
    """拡張機能のパスのリストを取得する。
    パスが複数指定されていてもひとつしか指定されていなくてもループできるようにする。
    """
    # 拡張機能の項目がなければNoneを返す。
    if 'extensions' not in config:
        return None
    # 目的の拡張機能のパスがあれば取得する。
    config_extensions_something = config.extensions.get(key)
    if config_extensions_something is None:
        return None
    if config_extensions_something == "":
        return None
    if isinstance(config_extensions_something, str):
        return [config_extensions_something]
    if isinstance(config_extensions_something, Iterable):
        return list(config_extensions_something)
    # 空文字列でもNULLでもリストでも文字列でもない場合
    raise TypeError(
        f'Extension path must be null or strings or list, not {type(config_extensions_something)} for {config_extensions_something}'
    )

def get_paths(temp_dir: str):
    # 各種出力ファイルのパスを設定
    path_temp_ust = abspath(join(temp_dir, 'temp.ust'))
    path_temp_table = abspath(join(temp_dir, 'temp.table'))
    path_full_score = abspath(join(temp_dir, 'score.full'))
    path_mono_score = abspath(join(temp_dir, 'score.lab'))
    path_full_timing = abspath(join(temp_dir, 'timing.full'))
    path_mono_timing = abspath(join(temp_dir, 'timing.lab'))
    path_acoustic = abspath(join(temp_dir, 'acoustic.csv'))
    path_f0 = abspath(join(temp_dir, 'f0.csv'))
    path_spectrogram = abspath(join(temp_dir, 'spectrogram.csv'))
    path_aperiodicity = abspath(join(temp_dir, 'aperiodicity.csv'))
    path_question = abspath(join(temp_dir, 'temp.hed'))
    return path_temp_ust, path_temp_table, \
        path_full_score, path_mono_score, \
        path_full_timing, path_mono_timing, \
        path_acoustic, path_f0, path_spectrogram, \
        path_aperiodicity, path_question

def set_config(config: DictConfig,model_dir: str,temp_dir: str):
    path_temp_ust, path_temp_table, \
    path_full_score, path_mono_score, \
    path_full_timing, path_mono_timing, \
    path_acoustic, path_f0, path_spectrogram, \
    path_aperiodicity, temp_path_question = get_paths(temp_dir)
    path_question = abspath(join(model_dir, 'qst.hed'))
    if isfile(temp_path_question):
        path_question = temp_path_question
    enconfigObj = {
        "verbose": 0,
        "model_dir": model_dir,
        "acoustic":{
            "in_scaler_path":"null",
            "out_scaler_path":"null",
            "model_yaml": abspath(join(model_dir, 'acoustic_model.yaml')),
            "checkpoint": abspath(join(model_dir, 'acoustic_model.pth'))
        },
        "question_path": path_question
    }
    ex_enconfig = OmegaConf.create(enconfigObj)
    config = OmegaConf.merge(config, ex_enconfig)
    return config

class SimpleEnunuServer(SPSVS):
    """SimpleEnunuで合成するとき用に、外部ソフトでタイミング補正ができるようにする。
    Args:
        model_dir (str): NNSVSのモデルがあるフォルダ
        device (str): 'cuda' or 'cpu'
    """
    def __init__(self,
            model_dir: str,
            temp_dir: str,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=0,
            **kwargs):
        super().__init__(model_dir, device=device, verbose=verbose, **kwargs)
        # self.voice_dir = None
        # self.path_plugin = None
        # self.path_ust = None
        self.path_full_score = None
        self.path_mono_score = None
        self.path_full_timing = None
        self.path_mono_timing = None
        self.path_acoustic = None
        self.path_f0 = None
        self.path_spcetrogram = None
        self.path_aperiodicity = None
        self.path_question = None
        
        # self.path_wav = None

    
    def set_paths(self, temp_dir):
        """ファイル入出力のPATHを設定する
        """

        path_temp_ust, path_temp_table, \
        self.path_full_score, self.path_mono_score, \
        self.path_full_timing, self.path_mono_timing, \
        self.path_acoustic, self.path_f0, self.path_spcetrogram, \
        self.path_aperiodicity, self.path_question = get_paths(temp_dir)

    def edit_timing(self, duration_modified_labels):
        """外部ツールでタイミング補正する
        """
        # タイミング補正ツールが指定されていない時はskip
        if 'extensions' not in self.config:
            return duration_modified_labels
        if self.config.extensions.get('timing_editor') is None:
            return duration_modified_labels
        # 外部ツールでmono_timingを編集
        self.logger.info(
            'Editing timing with %s',
            self.config.extensions.timing_editor
        )
        enulib.extensions.run_extension(
            self.config.extensions.timing_editor,
            full_score=self.path_full_score,
            mono_score=self.path_mono_score,
            full_timing=self.path_full_timing,
            mono_timing=self.path_mono_timing
        )
        # mono_timing の時刻情報を full_timing に移植
        enulib.extensions.merge_mono_time_change_to_full(
            self.path_mono_timing,
            self.path_full_timing
        )
        # 編集後のfull_timing を読み取る
        duration_modified_labels = hts.load(self.path_full_timing).round_()
        return duration_modified_labels

    def svs_timing(
        self,
        labels,
        vocoder_type='world',
        post_filter_type='gv',
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        vuv_threshold=0.5,
        style_shift=0,
        force_fix_vuv=False,
        fill_silence_to_rest=False,
        dtype=np.int16,
        peak_norm=False,
        loudness_norm=False,
        target_loudness=-20,
        segmented_synthesis=False,
        **kwargs
    ):
        """Synthesize waveform from HTS labels.
        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
            vocoder_type (str): Vocoder type. One of ``world``, ``pwg`` or ``usfgan``.
                If ``auto`` is specified, the vocoder is automatically selected.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
            trajectory_smoothing (bool): Whether to smooth acoustic feature trajectory.
            trajectory_smoothing_cutoff (int): Cutoff frequency for trajectory smoothing.
            trajectory_smoothing_cutoff_f0 (int): Cutoff frequency for trajectory
                smoothing of f0.
            vuv_threshold (float): Threshold for VUV.
            style_shift (int): style shift parameter
            force_fix_vuv (bool): Whether to correct VUV.
            fill_silence_to_rest (bool): Fill silence to rest frames.
            dtype (np.dtype): Data type of the output waveform.
            peak_norm (bool): Whether to normalize the waveform by peak value.
            loudness_norm (bool): Whether to normalize the waveform by loudness.
            target_loudness (float): Target loudness in dB.
            segmneted_synthesis (bool): Whether to use segmented synthesis.
        """
        start_time = time.time()
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ["world", "pwg", "usfgan", "auto"]:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        if post_filter_type not in ["merlin", "nnsvs", "gv", "none"]:
            raise ValueError(f"Unknown post-filter type: {post_filter_type}")
        # Predict timinigs
        duration_modified_labels = self.predict_timing(labels)
        # NOTE: ここにタイミング補正のための割り込み処理を追加-----------
        # mono_score を出力
        with open(self.path_mono_score, 'w', encoding='utf-8') as f:
            f.write(str(nnsvs.io.hts.full_to_mono(labels)))
        # mono_timing を出力
        with open(self.path_mono_timing, 'w', encoding='utf-8') as f:
            f.write(str(nnsvs.io.hts.full_to_mono(duration_modified_labels)))
        # full_timing を出力
        with open(self.path_full_timing, 'w', encoding='utf-8') as f:
            f.write(str(duration_modified_labels))


        return self.path_mono_score,self.path_mono_timing,self.path_full_timing
    

    def interp1d(self,f0=[],kind='linear'):
        ndim = f0.ndim
        if len(f0) != f0.size:
            raise RuntimeError("1d array is only supported")
        continuous_f0 = f0.flatten()
        nonzero_indices = np.where(continuous_f0 > 0)[0]

        # Nothing to do
        if len(nonzero_indices) <= 0:
            return f0

        # Need this to insert continuous values for the first/end silence segments
        continuous_f0[0] = continuous_f0[nonzero_indices[0]]
        continuous_f0[-1] = continuous_f0[nonzero_indices[-1]]

        # Build interpolation function
        nonzero_indices = np.where(continuous_f0 > 0)[0]
        if kind == "akima":
            interp_func = interpolate.Akima1DInterpolator(
                x=nonzero_indices, y=continuous_f0[continuous_f0 > 0])
        elif kind == "pchip":
            interp_func = interpolate.PchipInterpolator(
                x=nonzero_indices, y=continuous_f0[continuous_f0 > 0])
        else:
            interp_func = interpolate.interp1d(
            nonzero_indices, continuous_f0[continuous_f0 > 0], kind=kind)

        # Fill silence segments with interpolated values
        zero_indices = np.where(continuous_f0 <= 0)[0]
        continuous_f0[zero_indices] = interp_func(zero_indices)

        if ndim == 2:
            return continuous_f0[:, None]
        return continuous_f0


    def svs_acoustic(
        self,
        labels,
        vocoder_type='world',
        post_filter_type='gv',
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        vuv_threshold=0.5,
        style_shift=0,
        force_fix_vuv=False,
        fill_silence_to_rest=False,
        dtype=np.int16,
        peak_norm=False,
        loudness_norm=False,
        target_loudness=-20,
        segmented_synthesis=False,
        kind='linear',
        **kwargs
    ):
        """Synthesize waveform from HTS labels.
        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
            vocoder_type (str): Vocoder type. One of ``world``, ``pwg`` or ``usfgan``.
                If ``auto`` is specified, the vocoder is automatically selected.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
            trajectory_smoothing (bool): Whether to smooth acoustic feature trajectory.
            trajectory_smoothing_cutoff (int): Cutoff frequency for trajectory smoothing.
            trajectory_smoothing_cutoff_f0 (int): Cutoff frequency for trajectory
                smoothing of f0.
            vuv_threshold (float): Threshold for VUV.
            style_shift (int): style shift parameter
            force_fix_vuv (bool): Whether to correct VUV.
            fill_silence_to_rest (bool): Fill silence to rest frames.
            dtype (np.dtype): Data type of the output waveform.
            peak_norm (bool): Whether to normalize the waveform by peak value.
            loudness_norm (bool): Whether to normalize the waveform by loudness.
            target_loudness (float): Target loudness in dB.
            segmneted_synthesis (bool): Whether to use segmented synthesis.
        """
        start_time = time.time()
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ["world", "pwg", "usfgan", "auto"]:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        if post_filter_type not in ["merlin", "nnsvs", "gv", "none"]:
            raise ValueError(f"Unknown post-filter type: {post_filter_type}")
        
        # 編集後のfull_timing を読み取る
        duration_modified_labels = self.predict_timing(labels)
        # NOTE: ここにタイミング補正のための割り込み処理を追加-----------
        # mono_score を出力
        with open(self.path_mono_score, 'w', encoding='utf-8') as f:
            f.write(str(nnsvs.io.hts.full_to_mono(labels)))
        # mono_timing を出力
        with open(self.path_mono_timing, 'w', encoding='utf-8') as f:
            f.write(str(nnsvs.io.hts.full_to_mono(duration_modified_labels)))
        # full_timing を出力
        with open(self.path_full_timing, 'w', encoding='utf-8') as f:
            f.write(str(duration_modified_labels))
        
        # NOTE: segmented synthesis is not well tested. There MUST be better ways
        # to do this.
        if segmented_synthesis:
            self.logger.warning(
                "Segmented synthesis is not well tested. Use it on your own risk."
            )
            # NOTE: ここsegment_labels が nnsvs の中の関数にあるので呼び出せるように改造済み
            duration_modified_labels_segs = nnsvs.io.hts.segment_labels(
                duration_modified_labels,
                # the following parameters are based on experiments in the NNSVS's paper
                # tuned with Namine Ritsu's database
                silence_threshold=0.1,
                min_duration=1.9,
                force_split_threshold=5.0,
            )
            from tqdm.auto import tqdm
        else:
            duration_modified_labels_segs = [duration_modified_labels]

            def tqdm(x, **kwargs):
                return x

        # Run acoustic model and vocoder
        hts_frame_shift = int(self.config.frame_period * 1e4)
        datas,f0s, spectrograms, aperiodicitys = [],[],[],[]
        params = 0
        self.logger.info(
            f"Number of segments: {len(duration_modified_labels_segs)}")
        for duration_modified_labels_seg in tqdm(
            duration_modified_labels_segs,
            desc="[segment]",
            total=len(duration_modified_labels_segs),
        ):
            duration_modified_labels.frame_shift = hts_frame_shift

            # Predict acoustic features
            # NOTE: if non-zero pre_f0_shift_in_cent is specified, the input pitch
            # will be shifted before running the acoustic model
            acoustic_features = self.predict_acoustic(
                duration_modified_labels,
                f0_shift_in_cent=style_shift * 100,
            )

            # Post-processing for acoustic features
            # NOTE: if non-zero post_f0_shift_in_cent is specified, the output pitch
            # will be shifted as a part of post-processing
            multistream_features = self.postprocess_acoustic(
                acoustic_features=acoustic_features,
                duration_modified_labels=duration_modified_labels,
                trajectory_smoothing=trajectory_smoothing,
                trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
                trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
                force_fix_vuv=force_fix_vuv,
                fill_silence_to_rest=fill_silence_to_rest,
                f0_shift_in_cent=-style_shift * 100,
            )


            data,f0, spectrogram, aperiodicity = [],[],[],[]
            params = len(multistream_features)
            if params >= 4:
                mgc, lf0, vuv, bap = multistream_features
                # Generate WORLD parameters
                f0, spectrogram, aperiodicity = gen_world_params(
                    mgc, lf0, vuv, bap, self.sample_rate, vuv_threshold=vuv_threshold
                )
                f0s.append(f0)
                spectrograms.append(spectrogram)
                aperiodicitys.append(aperiodicity)
            elif params == 3:
                data = self.predict_waveform(
                    multistream_features=multistream_features,
                    vocoder_type=vocoder_type,
                    vuv_threshold=vuv_threshold,
                )
                data = data.astype(np.double)
                datas.append(data)
        
        if params == 3:
            data = np.concatenate(datas, axis=0).reshape(-1)
            f0, spectrogram, aperiodicity = pyworld.wav2world(data, self.sample_rate)
            f0 = self.interp1d(f0=f0,kind=kind)
        else:
            f0 = np.concatenate(f0s, axis=0).reshape(-1)
            spectrogram = np.concatenate(spectrograms, axis=0).reshape(-1)
            aperiodicity = np.concatenate(aperiodicitys, axis=0).reshape(-1)

        # csvファイルとしてf0の行列を出力
        for path, array in (
            (self.path_f0, f0),
            (self.path_spcetrogram, spectrogram),
            (self.path_aperiodicity, aperiodicity),
        ):
            if array is not None:
                np.savetxt(
                    path,
                    array,
                    fmt='%.16f',
                    delimiter=','
                )

    
    def svs(
        self,
        labels,
        vocoder_type='world',
        post_filter_type='gv',
        trajectory_smoothing=True,
        trajectory_smoothing_cutoff=50,
        trajectory_smoothing_cutoff_f0=20,
        vuv_threshold=0.5,
        style_shift=0,
        force_fix_vuv=False,
        fill_silence_to_rest=False,
        dtype=np.int16,
        peak_norm=False,
        loudness_norm=False,
        target_loudness=-20,
        segmented_synthesis=False,
        **kwargs
    ):
        """Synthesize waveform from HTS labels.
        Args:
            labels (nnmnkwii.io.hts.HTSLabelFile): HTS labels
            vocoder_type (str): Vocoder type. One of ``world``, ``pwg`` or ``usfgan``.
                If ``auto`` is specified, the vocoder is automatically selected.
            post_filter_type (str): Post-filter type. ``merlin``, ``gv`` or ``nnsvs``
                is supported.
            trajectory_smoothing (bool): Whether to smooth acoustic feature trajectory.
            trajectory_smoothing_cutoff (int): Cutoff frequency for trajectory smoothing.
            trajectory_smoothing_cutoff_f0 (int): Cutoff frequency for trajectory
                smoothing of f0.
            vuv_threshold (float): Threshold for VUV.
            style_shift (int): style shift parameter
            force_fix_vuv (bool): Whether to correct VUV.
            fill_silence_to_rest (bool): Fill silence to rest frames.
            dtype (np.dtype): Data type of the output waveform.
            peak_norm (bool): Whether to normalize the waveform by peak value.
            loudness_norm (bool): Whether to normalize the waveform by loudness.
            target_loudness (float): Target loudness in dB.
            segmneted_synthesis (bool): Whether to use segmented synthesis.
        """
        start_time = time.time()
        vocoder_type = vocoder_type.lower()
        if vocoder_type not in ["world", "pwg", "usfgan", "auto"]:
            raise ValueError(f"Unknown vocoder type: {vocoder_type}")
        if post_filter_type not in ["merlin", "nnsvs", "gv", "none"]:
            raise ValueError(f"Unknown post-filter type: {post_filter_type}")

        # Predict timinigs
        duration_modified_labels = self.predict_timing(labels)

        # NOTE: ここにタイミング補正のための割り込み処理を追加-----------
        # mono_score を出力
        with open(self.path_mono_score, 'w', encoding='utf-8') as f:
            f.write(str(nnsvs.io.hts.full_to_mono(labels)))
        # mono_timing を出力
        with open(self.path_mono_timing, 'w', encoding='utf-8') as f:
            f.write(str(nnsvs.io.hts.full_to_mono(duration_modified_labels)))
        # full_timing を出力
        with open(self.path_full_timing, 'w', encoding='utf-8') as f:
            f.write(str(duration_modified_labels))
        # 外部で加工した結果でタイミング情報を置換
        duration_modified_labels = self.edit_timing(duration_modified_labels)
        # ---------------------------------------------------------------

        # NOTE: segmented synthesis is not well tested. There MUST be better ways
        # to do this.
        if segmented_synthesis:
            self.logger.warning(
                "Segmented synthesis is not well tested. Use it on your own risk."
            )
            # NOTE: ここsegment_labels が nnsvs の中の関数にあるので呼び出せるように改造済み
            duration_modified_labels_segs = nnsvs.io.hts.segment_labels(
                duration_modified_labels,
                # the following parameters are based on experiments in the NNSVS's paper
                # tuned with Namine Ritsu's database
                silence_threshold=0.1,
                min_duration=5.0,
                force_split_threshold=5.0,
            )
            from tqdm.auto import tqdm
        else:
            duration_modified_labels_segs = [duration_modified_labels]

            def tqdm(x, **kwargs):
                return x

        # Run acoustic model and vocoder
        hts_frame_shift = int(self.config.frame_period * 1e4)
        wavs = []
        self.logger.info(
            f"Number of segments: {len(duration_modified_labels_segs)}")
        for duration_modified_labels_seg in tqdm(
            duration_modified_labels_segs,
            desc="[segment]",
            total=len(duration_modified_labels_segs),
        ):
            duration_modified_labels_seg.frame_shift = hts_frame_shift

            # Predict acoustic features
            # NOTE: if non-zero pre_f0_shift_in_cent is specified, the input pitch
            # will be shifted before running the acoustic model
            acoustic_features = self.predict_acoustic(
                duration_modified_labels_seg,
                f0_shift_in_cent=style_shift * 100,
            )

            # Post-processing for acoustic features
            # NOTE: if non-zero post_f0_shift_in_cent is specified, the output pitch
            # will be shifted as a part of post-processing
            multistream_features = self.postprocess_acoustic(
                acoustic_features=acoustic_features,
                duration_modified_labels=duration_modified_labels_seg,
                trajectory_smoothing=trajectory_smoothing,
                trajectory_smoothing_cutoff=trajectory_smoothing_cutoff,
                trajectory_smoothing_cutoff_f0=trajectory_smoothing_cutoff_f0,
                force_fix_vuv=force_fix_vuv,
                fill_silence_to_rest=fill_silence_to_rest,
                f0_shift_in_cent=-style_shift * 100,
            )

            # Generate waveform by vocoder
            wav = self.predict_waveform(
                multistream_features=multistream_features,
                vocoder_type=vocoder_type,
                vuv_threshold=vuv_threshold,
            )

            wavs.append(wav)

        # Concatenate segmented waveforms
        wav = np.concatenate(wavs, axis=0).reshape(-1)

        # Post-processing for the output waveform
        wav = self.postprocess_waveform(
            wav,
            dtype=dtype,
            peak_norm=peak_norm,
            loudness_norm=loudness_norm,
            target_loudness=target_loudness,
        )
        self.logger.info(f"Total time: {time.time() - start_time:.3f} sec")
        RT = (time.time() - start_time) / (len(wav) / self.sample_rate)
        self.logger.info(f"Total real-time factor: {RT:.3f}")
        return wav, self.sample_rate

def run_timing(config: DictConfig, temp_dir: str ,engine: SimpleEnunuServer):
    path_temp_ust, path_temp_table, \
    path_full_score, path_mono_score, \
    path_full_timing, path_mono_timing, \
    path_acoustic, path_f0, path_spectrogram, \
    path_aperiodicity, temp_path_question = get_paths(temp_dir)

    # フルラベル(score)生成----------------------------------------------------------
    converter = get_standard_function_config(config, 'ust_converter')
    # フルラベル生成をしない場合
    if converter is None:
        pass
    # ENUNUの組み込み機能でUST→LAB変換をする場合
    elif converter == 'built-in':
        print(f'{datetime.now()} : converting UST to score with built-in function')
        enulib.utauplugin2score.utauplugin2score(
            path_temp_ust,
            path_temp_table,
            path_full_score,
            strict_sinsy_style=False
        )
        # full_score から mono_score を生成
        enulib.common.full2mono(path_full_score, path_mono_score)
    # 外部ソフトでUST→LAB変換をする場合
    else:
        print(
            f'{datetime.now()} : converting UST to score with built-in function{converter}')
        enulib.extensions.run_extension(
            converter,
            ust=path_temp_ust,
            table=path_temp_table,
            full_score=path_full_score,
            mono_score=path_mono_score
        )

    # フルラベル(timing) を生成 score.full -> timing.full-----------------
    calculator = get_standard_function_config(config, 'timing_calculator')
    # duration計算をしない場合
    if calculator is None:
        print(f'{datetime.now()} : skipped acoustic calculation')
    # ENUNUの組み込み機能で計算する場合
    elif calculator == 'built-in':
        print(f'{datetime.now()} : calculating timing with built-in function')
        enulib.timing.score2timing(
            config,
            path_full_score,
            path_full_timing
        )
        # フルラベルからモノラベルを生成
        enulib.common.full2mono(path_full_timing, path_mono_timing)

    #nnsvsで計算する場合
    elif calculator == 'nnsvs':
        
        print(f'{datetime.now()} : calculate timings with nnsvs function')
        
        # フルラベルファイルを読み取る
        logging.info('Loading LAB')
        labels = hts.load(path_full_score)

        path_mono_score,path_mono_timing,path_full_timing = engine.svs_timing(
            labels=labels,
            dtype=np.float32,
            vocoder_type='auto',
            post_filter_type='gv',
            force_fix_vuv=True,
            segmented_synthesis=True,
        )
    # 外部ソフトで計算する場合
    else:
        print(f'{datetime.now()} : calculating timing with {calculator}')
        enulib.extensions.run_extension(
            calculator,
            ust=path_temp_ust,
            table=path_temp_table,
            full_score=path_full_score,
            mono_score=path_mono_score,
            full_timing=path_full_timing,
            mono_timing=path_mono_timing
        )
    extension_list = get_extension_path_list(config, 'timing_editor')
    if extension_list is not None:
        # 複数ツールのすべてについて処理実施する
        for path_extension in extension_list:
            print(f'{datetime.now()} : editing timing with {path_extension}')
            # 変更前のモノラベルを読んでおく
            with open(path_mono_timing, encoding='utf-8') as f:
                str_mono_old = f.read()
            enulib.extensions.run_extension(
                path_extension,
                ust=path_temp_ust,
                table=path_temp_table,
                full_score=path_full_score,
                mono_score=path_mono_score,
                full_timing=path_full_timing,
                mono_timing=path_mono_timing
            )
            # 変更後のモノラベルを読む
            with open(path_mono_timing, encoding='utf-8') as f:
                str_mono_new = f.read()
            # モノラベルの時刻が変わっていたらフルラベルに転写して、
            # そうでなければフルラベルの時刻をモノラベルに転写する。
            # NOTE: 歌詞は編集していないという前提で処理する。
            if enulib.extensions.str_has_been_changed(str_mono_old, str_mono_new):
                enulib.extensions.merge_mono_time_change_to_full(
                    path_mono_timing, path_full_timing)
            else:
                enulib.extensions.merge_full_time_change_to_mono(
                    path_full_timing, path_mono_timing)
                

    return path_full_timing, path_mono_timing

def run_acoustic(config: DictConfig, temp_dir: str,engine: SimpleEnunuServer):
    path_temp_ust, path_temp_table, \
    path_full_score, path_mono_score, \
    path_full_timing, path_mono_timing, \
    path_acoustic, path_f0, path_spectrogram, \
    path_aperiodicity, temp_path_question = get_paths(temp_dir)
    
    calculator = get_standard_function_config(config, 'acoustic_calculator')
    # 計算をしない場合
    if calculator is None:
        print(f'{datetime.now()} : skipped acoustic calculation')
    elif calculator == 'nnsvs':
        print(f'{datetime.now()} : calculating acoustic with built-in function')

        # フルラベルファイルを読み取る
        logging.info('Loading LAB')
        labels = hts.load(path_full_timing).round_()
        
        # acoustic のファイルから f0, spectrogram, aperiodicity のファイルを出力
        kind = get_standard_function_config(config, 'pitch_curve_interpolation')
        if kind is None:
            kind = 'linear'
        print(f'{datetime.now()} : pitch interpolation mode : {kind}')
        
        engine.svs_acoustic(
            labels=labels,
            dtype=np.float32,
            vocoder_type='auto',
            post_filter_type='gv',
            force_fix_vuv=True,
            segmented_synthesis=False,
            kind=kind,
        )
    elif calculator == 'built-in':
        print(
            f'{datetime.now()} : calculating acoustic with built-in function')
        # timing.full から acoustic.csv を作る。
        enulib.acoustic.timing2acoustic(
            config, path_full_timing, path_acoustic)
        # acoustic のファイルから f0, spectrogram, aperiodicity のファイルを出力
        enulib.world.acoustic2world(
            config,
            path_full_timing,
            path_acoustic,
            path_f0,
            path_spectrogram,
            path_aperiodicity
        )
    else:
        print(
            f'{datetime.now()} : calculating acoustic with {calculator}')
        enulib.extensions.run_extension(
            calculator,
            ust=path_temp_ust,
            table=path_temp_table,
            full_score=path_full_score,
            mono_score=path_mono_score,
            full_timing=path_full_timing,
            mono_timing=path_mono_timing,
            acoustic=path_acoustic,
            f0=path_f0,
            spectrogram=path_spectrogram,
            aperiodicity=path_aperiodicity
        )

    # 音響パラメータを加工: acoustic.csv -> acoustic.csv -------------------------
    extension_list = get_extension_path_list(config, 'acoustic_editor')
    if extension_list is not None:
        for path_extension in extension_list:
            print(f'{datetime.now()} : editing acoustic with {path_extension}')
            enulib.extensions.run_extension(
                path_extension,
                ust=path_temp_ust,
                table=path_temp_table,
                full_score=path_full_score,
                mono_score=path_mono_score,
                full_timing=path_full_timing,
                mono_timing=path_mono_timing,
                acoustic=path_acoustic,
                f0=path_f0,
                spectrogram=path_spectrogram,
                aperiodicity=path_aperiodicity
            )


    return path_acoustic, path_f0, path_spectrogram, path_aperiodicity


def run_synthesizer(config: DictConfig, temp_dir: str,out_wav_path: str,engine: SimpleEnunuServer):
    path_temp_ust, path_temp_table, \
    path_full_score, path_mono_score, \
    path_full_timing, path_mono_timing, \
    path_acoustic, path_f0, path_spectrogram, \
    path_aperiodicity, temp_path_question = get_paths(temp_dir)

    # WORLDを使って音声ファイルを生成: acoustic.csv -> <songname>.wav--------------
    synthesizer = get_standard_function_config(config, 'wav_synthesizer')

    # ここでは合成をしない場合
    if synthesizer is None:
        print(f'{datetime.now()} : skipped synthesizing WAV')

    elif synthesizer == 'built-in':
        print(f'{datetime.now()} : synthesizing WAV with built-in function')
        # WAVファイル出力
        enulib.world.world2wav(
            config,
            path_f0,
            path_spectrogram,
            path_aperiodicity,
            out_wav_path
        )

    # 別途指定するソフトで合成する場合
    else:
        print(f'{datetime.now()} : synthesizing WAV with {synthesizer}')
        enulib.extensions.run_extension(
            synthesizer,
            ust=path_temp_ust,
            table=path_temp_table,
            full_score=path_full_score,
            mono_score=path_mono_score,
            full_timing=path_full_timing,
            mono_timing=path_mono_timing,
            acoustic=path_acoustic,
            f0=path_f0,
            spectrogram=path_spectrogram,
            aperiodicity=path_aperiodicity
        )

    # 音声ファイルを加工: <songname>.wav -> <songname>.wav
    extension_list = get_extension_path_list(config, 'wav_editor')
    if extension_list is not None:
        for path_extension in extension_list:
            print(f'{datetime.now()} : editing WAV with {path_extension}')
            enulib.extensions.run_extension(
                path_extension,
                ust=path_temp_ust,
                table=path_temp_table,
                full_score=path_full_score,
                mono_score=path_mono_score,
                full_timing=path_full_timing,
                mono_timing=path_mono_timing,
                acoustic=path_acoustic,
                f0=path_f0,
                spectrogram=path_spectrogram,
                aperiodicity=path_aperiodicity
            )

    # print(f'{datetime.now()} : converting LAB to JSON')
    # hts2json(path_full_score, path_json)

    return out_wav_path



def setup(path_plugin: str):
    ""
    # USTの形式のファイルでなければエラー
    if not path_plugin.endswith('.tmp') or path_plugin.endswith('.ust'):
        raise ValueError('Input file must be UST or TMP(plugin).')
    # UTAUの一時ファイルに書いてある設定を読み取る
    logging.info('reading settings in TMP')
    path_ust, voice_dir, _ = get_project_path(path_plugin)

    # ENUNU<1.0.0 向けの構成のモデルな場合
    if exists(join(voice_dir, 'enuconfig.yaml')):
        logging.info(
            'Regacy ENUNU model is selected. Converting it for the compatibility...'
        )
        try:
                with open(join(voice_dir, 'enuconfig.yaml'), encoding='utf-8') as f:
                # configファイルを読み取る
                    print(f'{datetime.now()} : reading enuconfig')
                    config = DictConfig(OmegaConf.load(f))
                model_dir = join(voice_dir, 'model')
                makedirs(model_dir, exist_ok=True)
                print('----------------------------------------------')
                wrapped_enunu2nnsvs(voice_dir, model_dir,config)
                print('\n----------------------------------------------')
                logging.info('Converted.')
                table_path = config.table_path
        except Exception as e:
            print(e)
    # ENUNU=>1.0.0 または SimpleEnunu 用に作成されたNNSVSモデルの場合
    elif packed_model_exists(join(voice_dir, 'model')):
        model_dir = join(voice_dir, 'model')
        table_path = find_table(model_dir)
        try:
            with open(join(model_dir, 'config.yaml'), encoding='utf-8') as f:
            # configファイルを読み取る
                print(f'{datetime.now()} : reading enuconfig')
                config = DictConfig(OmegaConf.load(f))
                config = set_config(config,model_dir)
                logging.info('Converted.')
        except Exception as e:
            print(e)
    # ENUNU 用ではない通常のNNSVSモデルの場合
    elif packed_model_exists(voice_dir):
        model_dir = voice_dir
        table_path = find_table(model_dir)
        try:
            with open(join(model_dir, 'config.yaml'), encoding='utf-8') as f:
            # configファイルを読み取る
                print(f'{datetime.now()} : reading enuconfig')
                config = DictConfig(OmegaConf.load(f))
                config = set_config(config,model_dir)
                logging.info('Converted.')
        except Exception as e:
            print(e)
        logging.warning(
            'NNSVS model is selected. This model might be not ready for ENUNU.')

    # configファイルがあるか調べて、なければ例外処理
    else:
        raise Exception('UTAU音源選択でENUNU用モデルを指定してください。')
    assert model_dir
    assert table_path

    # カレントディレクトリを音源フォルダに変更する
    chdir(voice_dir)

    # 日付時刻を取得
    str_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 入出力パスを設定する
    if path_ust is not None:
        songname = splitext(basename(path_ust))[0]
        out_dir = dirname(path_ust)
        temp_dir = join(out_dir, f'{songname}_enutemp')
        path_wav = abspath(join(out_dir, f'{songname}__{str_now}.wav'))
    # WAV出力パス指定なしかつUST未保存の場合
    else:
        logging.info('USTが保存されていないので一時フォルダにWAV出力します。')
        songname = f'temp__{str_now}'
        out_dir = mkdtemp(prefix='enunu-')
        temp_dir = join(out_dir, f'{songname}_enutemp')
        path_wav = abspath(join(out_dir, f'{songname}__{str_now}.wav'))
    makedirs(temp_dir, exist_ok=True)

    # 各種出力ファイルのパスを設定
    # path_plugin = path_plugin
    path_temp_ust, path_temp_table, \
    path_full_score, path_mono_score, \
    path_full_timing, path_mono_timing, \
    path_acoustic, path_f0, path_spectrogram, \
    path_aperiodicity, temp_path_question = get_paths(temp_dir)
    

    # USTを一時フォルダに複製
    print(f'{datetime.now()} : copying UST')
    copy(path_plugin, path_temp_ust)
    print(f'{datetime.now()} : copying Table')
    copy(table_path, path_temp_table)

    # モデルを読み取る
    logging.info('Loading models')
    engine = SimpleEnunuServer(
    model_dir,temp_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
    engine.set_paths(temp_dir=temp_dir)
    
    return config, temp_dir,engine


def main(path_plugin: str, path_wav: Union[str, None] = None, play_wav=True) -> str:
    """
    UTAUプラグインのファイルから音声を生成する
    """
    # USTの形式のファイルでなければエラー
    if not path_plugin.endswith('.tmp') or path_plugin.endswith('.ust'):
        raise ValueError('Input file must be UST or TMP(plugin).')
    # UTAUの一時ファイルに書いてある設定を読み取る
    logging.info('reading settings in TMP')
    path_ust, voice_dir, _ = get_project_path(path_plugin)

    # ENUNU=>1.0.0 または SimpleEnunu 用に作成されたNNSVSモデルの場合
    if packed_model_exists(join(voice_dir, 'model')):
        model_dir = join(voice_dir, 'model')
    # ENUNU 用ではない通常のNNSVSモデルの場合
    elif packed_model_exists(voice_dir):
        model_dir = voice_dir
        logging.warning(
            'NNSVS model is selected. This model might be not ready for ENUNU.')

    # ENUNU<1.0.0 向けの構成のモデルな場合
    elif exists(join(voice_dir, 'enuconfig.yaml')):
        logging.info(
            'Regacy ENUNU model is selected. Converting it for the compatibility...'
        )
        model_dir = join(voice_dir, 'model')
        makedirs(model_dir, exist_ok=True)
        print('----------------------------------------------')
        wrapped_enunu2nnsvs(voice_dir, model_dir)
        print('\n----------------------------------------------')
        logging.info('Converted.')

    # configファイルがあるか調べて、なければ例外処理
    else:
        raise Exception('UTAU音源選択でENUNU用モデルを指定してください。')
    assert model_dir

    # カレントディレクトリを音源フォルダに変更する
    chdir(voice_dir)

    # 日付時刻を取得
    str_now = datetime.now().strftime('%Y%m%d_%H%M%S')

    # wav出力パスが指定されていない(プラグインとして実行している)場合
    if path_wav is None:
        # 入出力パスを設定する
        if path_ust is not None:
            songname = splitext(basename(path_ust))[0]
            out_dir = dirname(path_ust)
            temp_dir = join(out_dir, f'{songname}_enutemp')
            path_wav = abspath(join(out_dir, f'{songname}__{str_now}.wav'))
        # WAV出力パス指定なしかつUST未保存の場合
        else:
            logging.info('USTが保存されていないので一時フォルダにWAV出力します。')
            songname = f'temp__{str_now}'
            out_dir = mkdtemp(prefix='enunu-')
            temp_dir = join(out_dir, f'{songname}_enutemp')
            path_wav = abspath(join(out_dir, f'{songname}__{str_now}.wav'))
    # WAV出力パスが指定されている場合
    else:
        songname = splitext(basename(path_wav))[0]
        out_dir = dirname(path_wav)
        temp_dir = join(out_dir, f'{songname}_enutemp')
        path_wav = abspath(path_wav)
    makedirs(temp_dir, exist_ok=True)
    path_full_score = join(temp_dir, f'{songname}_score.full')

    # モデルを読み取る
    logging.info('Loading models')
    engine = SimpleEnunuServer(
        model_dir, device='cuda' if torch.cuda.is_available() else 'cpu')
    engine.set_paths(temp_dir=temp_dir, songname=songname)

    # UST → LAB の変換をする
    logging.info('Converting UST -> LAB')
    enulib.utauplugin2score.utauplugin2score(
        path_plugin,
        find_table(model_dir),
        path_full_score,
        strict_sinsy_style=False
    )

    # フルラベルファイルを読み取る
    logging.info('Loading LAB')
    labels = hts.load(path_full_score)

    # 音声を生成する
    logging.info('Generating WAV')
    wav_data, sample_rate = engine.svs(
        labels,
        dtype=np.float32,
        vocoder_type='auto',
        post_filter_type='gv',
        force_fix_vuv=True,
        segmented_synthesis=True,
    )

    # wav出力のフォーマットを確認する
    print(sample_rate)

    wav_data = adjust_wav_gain_for_float32(wav_data)
    wavfile.write(path_wav, rate=sample_rate, data=wav_data)

    # 音声を再生する。
    if exists(path_wav) and play_wav is True:
        startfile(path_wav)

    return path_wav


if __name__ == '__main__':
    logging.debug('sys.argv: %s', sys.argv)
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1], None)
    elif len(sys.argv) == 1:
        main(input('Input file path of TMP(plugin)\n>>> ').strip('"'), None)
    else:
        raise Exception('引数が多すぎます。/ Too many arguments.')
