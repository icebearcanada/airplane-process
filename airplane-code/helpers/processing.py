import h5py
import numba
import numpy as np
import ctypes as C
import digital_rf
import os
from . import dsp

def load_cuda(config):
    if not config.cuda:
        try:
            import cupy as xp
        except ModuleNotFoundError:
            import numpy as xp
            print("cupy module not found, using numpy instead.")
    else:
        import numpy as xp
    return xp

def ssmfx(meas0, meas1, code, averages, nrang, fdec, codelen, clutter_gates):
    """
    Formats measured data and CUDA function inputs and calls wrapped function for determining the cross-correlation spectra of
    selected antenna pair.

    Args:
        meas0 (complex64 np.array): First antenna voltages loaded from HDF5 with phase and magnitude corrections.
        meas1 (complex64 np.array): Second antenna voltages loaded from HDF5 with phase and magnitude corrections.
        code (float32 np.array): Transmitted psuedo-random code sequence.
        averages (int): The number of 0.1 second averages to be performed on the GPU.
        nrang (int): Number of range gates being processed. Nominally 2000.
        fdec (int): Decimation rate to be used by GPU processing, effects Doppler resolution. Nominally 200.
        codelen (int): Length of the transmitted psuedo-random code sequence.

    Returns:
        S (complex64 np.array): 2D Spectrum output for antenna pair (Doppler shift x Range).

    Notes:
        * ssmf.cu could be modified to allow code to be a float32 input. This would reduce memory requirements
          on the GPU.
        * 'check' input of __fmed is 1 which indicates a pair of antennas is being processed.
    """
    nfreq = int(codelen / fdec)
    result_size = nfreq * nrang
    code = code.astype(np.complex64)
    result = np.zeros((nrang, nfreq), dtype=np.complex64)
    variance = np.zeros((nrang, nfreq), dtype=np.complex64)
    # Create pointers to convert python tpyes to C types
    m_p0 = meas0.ctypes.data_as(C.POINTER(C.c_float))
    m_p1 = meas1.ctypes.data_as(C.POINTER(C.c_float))
    c_p = code.ctypes.data_as(C.POINTER(C.c_float))
    r_p = result.ctypes.data_as(C.POINTER(C.c_float))
    v_p = variance.ctypes.data_as(C.POINTER(C.c_float))
    # Runs ssmf.cu on data set using defined pointers
    __fmed(m_p0, m_p1, c_p, r_p, v_p, len(meas0), codelen, result_size, averages, 1)

    return result, variance


def ssmfx_cupy(v0, v1, code, navg, nrng, fdec, codelen, clutter_gates):
    """
    Formats measured data and CUDA function inputs and calls wrapped function for determining the cross-correlation spectra of
    selected antenna pair.

    Args:
        v0 (complex64 xp.array): First antenna voltages loaded from HDF5 with phase and magnitude corrections.
        v1 (complex64 xp.array): Second antenna voltages loaded from HDF5 with phase and magnitude corrections.
        code (float32 xp.array): Transmitted psuedo-random code sequence.
        navg (int): The number of 0.1 second averages to be performed on the GPU.
        nrng (int): Number of range gates being processed. Nominally 2000.
        fdec (int): Decimation rate to be used by GPU processing, effects Doppler resolution. Nominally 200.
        codelen (int): Length of the transmitted psuedo-random code sequence.

    Returns:
        S (complex64 np.array): 2D Spectrum output for antenna pair (Doppler shift x Range).
    """
    import cupy as xp
    # todo: improve the integer casting
    nfreq = int(codelen / fdec)
    code = code.astype(xp.complex64)
    #v0_filtered = dsp.unmatched_filtering(v0, code, int(codelen), int(nrng), int(fdec), int(navg))
    #v1_filtered = dsp.unmatched_filtering(v1, code, int(codelen), int(nrng), int(fdec), int(navg))
    #spectra, variance = dsp.wiener_khinchin(v0_filtered, v1_filtered, navg)
    spectra, variance = dsp.wiener_khinchin(dsp.unmatched_filtering(v0, code, int(codelen), int(nrng), int(fdec), int(navg)), dsp.unmatched_filtering(v1, code, int(codelen), int(nrng), int(fdec), int(navg)), navg)

    return spectra, variance

def ssmfx_cupy_v2(v0, v1, code, navg, nrng, fdec, codelen, clutter_gates):
    """
    Formats measured data and CUDA function inputs and calls wrapped function for determining the cross-correlation spectra of
    selected antenna pair.

    Args:
        v0 (complex64 xp.array): First antenna voltages loaded from HDF5 with phase and magnitude corrections.
        v1 (complex64 xp.array): Second antenna voltages loaded from HDF5 with phase and magnitude corrections.
        code (float32 xp.array): Transmitted psuedo-random code sequence.
        navg (int): The number of 0.1 second averages to be performed on the GPU.
        nrng (int): Number of range gates being processed. Nominally 2000.
        fdec (int): Decimation rate to be used by GPU processing, effects Doppler resolution. Nominally 200.
        codelen (int): Length of the transmitted psuedo-random code sequence.

    Returns:
        S (complex64 np.array): 2D Spectrum output for antenna pair (Doppler shift x Range).
    """
    import cupy as xp

    nfreq = int(codelen / fdec)
    code = code.astype(xp.complex64)

    variance_samples = xp.zeros((int(nrng),int(nfreq), int(navg)))
    for i in range(int(navg)):
        variance_samples[:, :, i] = dsp.wiener_khinchin_v2(dsp.unmatched_filtering_v2(v0[int(codelen)*i:int(codelen)*(i+1) + int(nrng)], code, int(codelen), int(nrng), int(fdec), int(navg)), dsp.unmatched_filtering_v2(v1[int(codelen)*i:int(codelen)*(i+1) + int(nrng)], code, int(codelen), int(nrng), int(fdec), int(navg)), int(navg))

    spectra = xp.sum(variance_samples, axis=2)/int(navg)
    re = xp.sqrt(xp.sum((xp.real(variance_samples) - xp.real(spectra[:, :, xp.newaxis])) * (xp.real(variance_samples) - xp.real(spectra[:, :, xp.newaxis])), axis=2) / int(navg))
    im = xp.sqrt(xp.sum((xp.imag(variance_samples) - xp.imag(spectra[:, :, xp.newaxis])) * (xp.imag(variance_samples) - xp.imag(spectra[:, :, xp.newaxis])), axis=2) / int(navg))
    variance = re + 1j*im
    return spectra, variance


def decx(config, time, data, bcode, channel1, channel2, correction1, correction2):
    """
    Performs cross-correlation and decimation for inputed baseline from the radar data

    Parameters
    ----------


    Returns
    -------


    Notes
    -----
        * ssmfx CUDA can only handle number_ranges = 2000 exactly. For farther ranges we loop at step size 2000.
        * Currently the rea_vector command is resulting in an error at the end of execution. This oes not appear to
          affect the output of the script. Issue may be in h5py or digital_rf. This error only appears when using python3
    """
    
    xp = load_cuda(config)
    if config.number_ranges <= 2000:
        start_sample = int(time * config.raw_sample_rate) - config.timestamp_corr
        step_sample = config.code_length * config.incoherent_averages + config.number_ranges
        try:
            data1 = xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel1) * correction1)
            data2 = xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel2) * correction2)
            if not config.cuda:
                result, variance = ssmfx_cupy_v2(data1, data2, xp.asarray(bcode), xp.asarray(config.incoherent_averages),
                                          config.number_ranges, xp.asarray(config.decimation_rate),
                                          xp.asarray(config.code_length), xp.asarray(config.clutter_gates))
            else:
                result, variance = ssmfx(data1, data2, xp.asarray(bcode), xp.asarray(config.incoherent_averages),
                                          config.number_ranges, xp.asarray(config.decimation_rate),
                                          xp.asarray(config.code_length), xp.asarray(config.clutter_gates))
            return xp.transpose(result), xp.transpose(variance)
        except IOError:
            print(f'Read number went beyond existing channels({channel1}, {channel2}) or data '
                  f'(start {start_sample}, step {step_sample}) and raised an IOError')
            return 1, 1

    else:
        start_sample = int(time * config.raw_sample_rate) - config.timestamp_corr
        step_sample = config.code_length * config.incoherent_averages + 2000
        try:
            data1 = dsp.calibration_correction(xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel1)),
                                               xp.asarray(correction1))
            data2 = dsp.calibration_correction(xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel2)),
                                               xp.asarray(correction2))
            if not config.cuda:
                result, variance = ssmfx_cupy_v2(data1, data2, xp.asarray(bcode), xp.asarray(config.incoherent_averages),
                                          config.number_ranges, xp.asarray(config.decimation_rate),
                                          xp.asarray(config.code_length), xp.asarray(config.clutter_gates))
            else: 
                result, variance = ssmfx(data1, data2, xp.asarray(bcode), xp.asarray(config.incoherent_averages), 2000,
                                          xp.asarray(config.decimation_rate), xp.asarray(config.code_length))
            for i in range(2000, config.number_ranges, 2000):
                try:
                    start_sample = int(time * config.raw_sample_rate) + i - config.timestamp_corr
                    data1 = dsp.calibration_correction(
                        xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel1)), xp.asarray(correction1))
                    data2 = dsp.calibration_correction(
                        xp.asarray(data.read_vector_c81d(start_sample, step_sample, channel2)), xp.asarray(correction2))
                    if not config.cuda:
                        r, v = ssmfx_cupy_v2(data1, data2, xp.asarray(bcode), config.incoherent_averages, 2000,
                                      config.decimation_rate, config.code_length)
                    else:
                        r, v = ssmfx(data1, data2, xp.asarray(bcode), config.incoherent_averages, 2000,
                                      config.decimation_rate, config.code_length)

                    result = xp.append(result, r, axis=0)
                    variance = xp.append(variance, v, axis=0)
                except IOError:
                    print(f'Read number went beyond existing channels({channel1}, {channel2}) or data '
                          f'(start {start_sample}, step {step_sample}) and raised an IOError')
                    return 1, 1
            return xp.transpose(result), xp.transpose(variance)
        except IOError:
            print(f'Read number went beyond existing channels({channel1}, {channel2}) or data '
                  f'(start {start_sample}, step {step_sample}) and raised an IOError')
            return 1, 1
