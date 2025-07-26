import numpy as np
import kaldi_native_fbank as knf
import json
import os
import tensorflow as tf
import threading

from models.base import BaseEmbeddingModel
from utils import getenv_required


def load_fbank_opts(path):
    with open(path) as f:
        features_config = json.load(f)
    opts = knf.FbankOptions()
    if "sample-frequency" in features_config:
        opts.frame_opts.samp_freq = features_config["sample-frequency"]
    if "frame-length" in features_config:
        opts.frame_opts.frame_length_ms = features_config["frame-length"]
    if "frame-shift" in features_config:
        opts.frame_opts.frame_shift_ms = features_config["frame-shift"]
    if "preemphasis-coefficient" in features_config:
        opts.frame_opts.preemph_coeff = features_config["preemphasis-coefficient"]
    if "remove-dc-offset" in features_config:
        opts.frame_opts.remove_dc_offset = features_config["remove-dc-offset"]
    if "dither" in features_config:
        opts.frame_opts.dither = features_config["dither"]
    if "window-type" in features_config:
        opts.frame_opts.window_type = features_config["window-type"]
    if "blackman-coeff" in features_config:
        opts.frame_opts.blackman_coeff = features_config["blackman-coeff"]
    if "round-to-power-of-two" in features_config:
        opts.frame_opts.round_to_power_of_two = features_config["round-to-power-of-two"]
    if "snip-edges" in features_config:
        opts.frame_opts.snip_edges = features_config["snip-edges"]
    if "num-mel-bins" in features_config:
        opts.mel_opts.num_bins = features_config["num-mel-bins"]
    if "low-freq" in features_config:
        opts.mel_opts.low_freq = features_config["low-freq"]
    if "high-freq" in features_config:
        opts.mel_opts.high_freq = features_config["high-freq"]
    if "vtln-low" in features_config:
        opts.mel_opts.vtln_low = features_config["vtln-low"]
    if "vtln-high" in features_config:
        opts.mel_opts.vtln_high = features_config["vtln-high"]
    if "debug-mel" in features_config:
        opts.mel_opts.debug_mel = features_config["debug-mel"]
    if "use-energy" in features_config:
        opts.use_energy = features_config["use-energy"]
    if "energy-floor" in features_config:
        opts.energy_floor = features_config["energy-floor"]
    if "raw-energy" in features_config:
        opts.raw_energy = features_config["raw-energy"]
    if "htk-compat" in features_config:
        opts.htk_compat = features_config["htk-compat"]
    if "use-log-fbank" in features_config:
        opts.use_log_fbank = features_config["use-log-fbank"]
    if "use-power" in features_config:
        opts.use_power = features_config["use-power"]
    return opts


class YandexEcapaTdnnEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, sample_rate: int, model_path: str):
        super().__init__(sample_rate)
        self.lock = threading.Lock()
        self.fbank_opts = load_fbank_opts(os.path.join(model_path, "features_config.json"))
        if self.fbank_opts.frame_opts.samp_freq != sample_rate:
            raise ValueError("Sample rate mismatch")
        self.head = tf.lite.Interpreter(model_path=os.path.join(model_path, "head.tflite"))
        self.body = tf.lite.Interpreter(model_path=os.path.join(model_path, "body.tflite"))

    def extract_embeddings(self, samples: np.ndarray) -> np.ndarray | None:
        self.lock.acquire()
        try:
            self.body.allocate_tensors()
            body_frame_length = self.body.get_input_details()[0]["shape"][1]

            fbank = knf.OnlineFbank(self.fbank_opts)
            fbank.accept_waveform(self.sample_rate, samples.tolist())
            partial_results = []
            for i in range(0, fbank.num_frames_ready, body_frame_length):
                frames = []
                for j in range(i, min(i + body_frame_length, fbank.num_frames_ready)):
                    frames.append(fbank.get_frame(j))
                if len(frames) != body_frame_length:
                    continue
                frames = np.array([np.array(frames)])
                self.body.set_tensor(self.body.get_input_details()[0]["index"], frames)
                self.body.invoke()
                body_output = self.body.get_tensor(self.body.get_output_details()[0]["index"])
                for part in body_output[0]:
                    partial_results.append(part)
            partial_results = np.array([np.array(partial_results)])

            self.head.resize_tensor_input(0, partial_results.shape, True)
            self.head.allocate_tensors()
            self.head.set_tensor(self.head.get_input_details()[0]["index"], partial_results)
            self.head.invoke()
            return self.head.get_tensor(self.head.get_output_details()[0]["index"])[0]
        finally:
            self.lock.release()


def create_yandex_ecapa_tdnn_embedding_model_from_env(sample_rate: int) -> BaseEmbeddingModel:
    return YandexEcapaTdnnEmbeddingModel(sample_rate, getenv_required("YANDEX_ECAPA_TDNN_MODEL_PATH"))
