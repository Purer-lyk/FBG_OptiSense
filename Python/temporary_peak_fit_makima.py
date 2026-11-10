"""Five or seven measured points -> modified-Akima cubic peak estimate.

The realtime route contains five legacy or seven measured points per grating.
A symmetric Gaussian forces both tails toward one constant baseline and can
therefore draw artificial platforms when the real FBG peak is asymmetric.
This module instead uses a modified Akima (Makima) piecewise-cubic interpolant:
it passes through every physical ADC sample, permits different left/right
flanks, and selects the global engineering-dBm maximum on the displayed curve grid.
The grid is 401 uniform points plus measured wavelengths (no extrapolation);
this is a grid approximation, with the first wavelength on ties, not a
derivative-root estimate. Interpolated points are not measured training inputs.

No peak-quality score or Gaussian proxy is calculated. The stored
wavelength axis remains the existing board calibration, not live wavelength-
meter truth.
"""



import numpy as np
from scipy.interpolate import Akima1DInterpolator
from temporary_optical_units import adc_code_to_dbm, transimpedance_ohm


FIT_METHOD = 'makima_piecewise_cubic_5point'
_RENDER_POINT_COUNT = 401


def _half_height_width(curve_x, curve_y, center_nm, baseline_adc, peak_adc):
    """Return effective FWHM and whether either crossing left the 5-point window."""

    half_level = float(baseline_adc + 0.5 * (peak_adc - baseline_adc))
    center_index = int(np.argmin(np.abs(curve_x - center_nm)))

    left_candidates = np.flatnonzero(curve_y[:center_index + 1] <= half_level)
    right_candidates = np.flatnonzero(curve_y[center_index:] <= half_level)
    limited = not len(left_candidates) or not len(right_candidates)

    if len(left_candidates):
        left_low = int(left_candidates[-1])
        left_high = min(left_low + 1, center_index)
        if left_low == left_high or curve_y[left_high] == curve_y[left_low]:
            left_nm = float(curve_x[left_low])
        else:
            left_nm = float(np.interp(
                half_level,
                [curve_y[left_low], curve_y[left_high]],
                [curve_x[left_low], curve_x[left_high]],
            ))
    else:
        left_nm = float(curve_x[0])

    if len(right_candidates):
        right_high = center_index + int(right_candidates[0])
        right_low = max(center_index, right_high - 1)
        if right_low == right_high or curve_y[right_low] == curve_y[right_high]:
            right_nm = float(curve_x[right_high])
        else:
            # The right flank is descending; reverse both arrays for np.interp.
            right_nm = float(np.interp(
                half_level,
                [curve_y[right_high], curve_y[right_low]],
                [curve_x[right_high], curve_x[right_low]],
            ))
    else:
        right_nm = float(curve_x[-1])

    return max(0.0, right_nm - left_nm), bool(limited)





def fit_five_points(wavelength_nm, adc, *, feedback_selector=0, adc_channel=1):
    """Compatibility wrapper for saved five-point routes."""
    return fit_peak_points(wavelength_nm, adc, expected_count=5,
                           feedback_selector=feedback_selector, adc_channel=adc_channel)


def fit_peak_points(wavelength_nm, adc, *, expected_count=None,
                    feedback_selector=0, adc_channel=1):
    count = len(wavelength_nm)
    point_label = '五点' if count == 5 else '七点'
    method = f'makima_piecewise_cubic_{count}point'
    def unavailable(reason):
        return dict(

            fit_available=False,

            reason=f'无法拟合：{reason}',
            center_nm=None,

            fit_method=method,
            input_point_count=count,
            optical_accuracy_verified=False,
        )

    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(adc, dtype=float)
    if (count not in (5, 7) or (expected_count is not None and count != expected_count)
            or x.shape != (count,) or y.shape != (count,)
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        return unavailable('数据不完整')
    if np.any(y < 0):
        return unavailable('ADC无效')



    order = np.argsort(x, kind='stable')
    if not np.array_equal(order, np.arange(count)):
        x, y = x[order], y[order]

    if np.any(np.diff(x) <= 0):
        return unavailable('波长坐标重复或无效')

    height = float(np.ptp(y))
    # CH2/CH3 have fixed 2 kΩ front ends; CH0/CH1 use the frame selector.
    channel = int(adc_channel)
    if channel not in range(4):
        return unavailable('ADC通道无效')
    selector = int(feedback_selector) if channel < 2 else 0
    try:
        measured_dbm = adc_code_to_dbm(y, selector, floor_nonpositive=True)
    except ValueError:
        return unavailable('模拟跨阻档位无效')


    if height <= 1e-12:
        dense_x = np.unique(np.concatenate((
            np.linspace(x[0], x[-1], _RENDER_POINT_COUNT), x
        )))
        dense_y = np.full_like(dense_x, float(np.mean(y)))
        baseline_adc = float(np.mean(y))
        fwhm_nm = float(x[-1] - x[0])
        width_limited = True
    else:
        try:
            spline = Akima1DInterpolator(
                x, y, method='makima', extrapolate=False
            )
            dense_x = np.unique(np.concatenate((
                np.linspace(x[0], x[-1], _RENDER_POINT_COUNT), x,
            )))
            dense_y = np.asarray(spline(dense_x), dtype=float)
            if not np.all(np.isfinite(dense_y)):
                raise FloatingPointError('Makima输出包含非有限值')
            baseline_adc = max(0.0, float(min(y[0], y[-1])))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return unavailable('三次样条计算失败')

    curve_dbm = adc_code_to_dbm(dense_y, selector, floor_nonpositive=True)
    curve_apex = int(np.argmax(curve_dbm))
    center_nm = float(dense_x[curve_apex])
    peak_adc = float(dense_y[curve_apex])
    if height > 1e-12:
        fwhm_nm, width_limited = _half_height_width(
            dense_x, dense_y, center_nm, baseline_adc, peak_adc
        )

    amplitude_adc = max(0.0, peak_adc - baseline_adc)
    effective_sigma_nm = float(
        fwhm_nm / 2.354820045 if fwhm_nm > 0 else 0.0
    )

    reason = f'{point_label}Makima曲线全局最大dBm·网格近似·精度未验证'
    return dict(

        fit_available=True,

        reason=reason,
        center_nm=float(center_nm),
        sigma_nm=effective_sigma_nm,
        fwhm_nm=float(fwhm_nm),
        fwhm_limited_by_five_point_window=bool(width_limited),
        fwhm_limited_by_sampling_window=bool(width_limited),
        baseline_adc=float(baseline_adc),
        amplitude_adc=float(amplitude_adc),
        peak_adc=float(peak_adc),
        peak_dbm=float(curve_dbm[curve_apex]),
        peak_selection_method='makima_curve_global_max_dbm',
        peak_source='interpolated',
        peak_curve_index=curve_apex,
        peak_grid_uniform_point_count=_RENDER_POINT_COUNT,
        peak_grid_step_nm=float((x[-1] - x[0]) / (_RENDER_POINT_COUNT - 1)),
        peak_grid_point_count=len(dense_x),
        peak_grid_approximation=True,
        peak_tie_rule='first_in_wavelength_order',
        feedback_selector=selector,
        adc_channel=channel,
        transimpedance_ohm=float(transimpedance_ohm(selector)),
        dbm_is_engineering_estimate=True,

        curve_x_nm=dense_x.tolist(),
        curve_adc=dense_y.tolist(),
        fit_method=method,
        input_point_count=count,
        optical_accuracy_verified=False,
    )


def fit_temporary_frame(rows, values, warmed_up=True, *,
                        feedback_selector=0, adc_channel=1):
    from temporary_test_mode import points_per_peak_for_rows
    count = points_per_peak_for_rows(rows)
    if (len(rows) != len(values) or len(rows) not in range(count, 9 * count + 1, count)):
        raise ValueError('实时拟合需要1～9个完整五点或七点峰')
    # warmed_up is retained only for old callers; it is not a peak-quality gate.
    return [fit_peak_points(
                [r['measured_nm'] for r in rows[i:i + count]],
                values[i:i + count],
                feedback_selector=feedback_selector, adc_channel=adc_channel,
            )
            for i in range(0, len(rows), count)]

