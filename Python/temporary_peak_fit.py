"""Five measured points -> bounded Gaussian+constant estimate, never repaired data.

Every finite five-point arrangement gets a numerical fit. Residuals belong
to the numerical fit only; no peak-quality verdict is produced.
The wavelength axis is the stored calibration, not live wavelength-meter truth.
This estimate is not an optical accuracy certification.
"""
import numpy as np


def fit_five_points(wavelength_nm, adc):
    def unavailable(reason):
        return dict(

            fit_available=False,

            reason=f'无法拟合：{reason}',
            center_nm=None,
            r_squared=None,
            optical_accuracy_verified=False,
        )

    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(adc, dtype=float)
    if x.shape != (5,) or y.shape != (5,) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return unavailable('数据不完整')
    if np.any(y < 0):
        return unavailable('ADC无效')



    order = np.argsort(x, kind='stable')
    if not np.array_equal(order, np.arange(5)):
        x, y = x[order], y[order]

    if np.any(np.diff(x) <= 0):
        return unavailable('波长坐标重复或无效')

    height = float(np.ptp(y))
    apex = int(np.argmax(y))

    origin = x[2]
    scale = float(np.median(np.diff(x)))
    u = (x-origin)/scale
    low, high = u[0]+.25, u[-1]-.25
    sigma_low, sigma_high = .35, float(u[-1]-u[0])

    def solve(centers, widths):
        centers, widths = np.meshgrid(centers, widths, indexing='ij')
        centers, widths = centers.ravel(), widths.ravel()
        g = np.exp(-.5*((u[None, :]-centers[:, None])/widths[:, None])**2)
        centered = g-g.mean(axis=1)[:, None]
        amplitude = np.sum(centered*(y-y.mean()), axis=1)/np.maximum(np.sum(centered**2, axis=1), 1e-12)
        baseline = y.mean()-amplitude*g.mean(axis=1)
        # Nonnegative baseline least squares boundary; no changes to raw y.
        negative = baseline < 0
        baseline[negative] = 0
        amplitude[negative] = np.sum(g[negative]*y, axis=1)/np.sum(g[negative]**2, axis=1)
        prediction = baseline[:, None]+amplitude[:, None]*g
        sse = np.sum((prediction-y)**2, axis=1)
        sse[(amplitude < 0) | ~np.isfinite(sse)] = np.inf
        best = int(np.argmin(sse))
        return centers[best], widths[best], amplitude[best], baseline[best], sse[best]

    if height <= 1e-12:
        center = float(u[2])
        sigma = max(sigma_low, float(u[-1] - u[0]) / 4.0)
        amplitude = 0.0
        baseline = float(np.mean(y))
        sse = 0.0
    else:
        center_grid = np.linspace(low, high, 61)
        width_grid = np.linspace(sigma_low, sigma_high, 32)
        coarse = solve(center_grid, width_grid)
        dc, ds = center_grid[1]-center_grid[0], width_grid[1]-width_grid[0]
        center, sigma, amplitude, baseline, sse = solve(
            np.linspace(max(low, coarse[0]-dc), min(high, coarse[0]+dc), 41),
            np.linspace(max(sigma_low, coarse[1]-ds), min(sigma_high, coarse[1]+ds), 41))
        if not np.all(np.isfinite([center, sigma, amplitude, baseline, sse])):
            center = float(u[apex] if height > 0 else u[2])
            sigma = max(sigma_low, float(u[-1] - u[0]) / 4.0)
            baseline = max(0.0, float(np.min(y)))
            amplitude = max(0.0, float(np.max(y) - baseline))
            prediction = baseline + amplitude * np.exp(
                -.5 * ((u - center) / sigma) ** 2
            )
            sse = float(np.sum((prediction - y) ** 2))


    rmse = float(np.sqrt(sse/5))
    variance = float(np.sum((y-y.mean())**2))
    r2 = 0.0 if variance <= 1e-12 else float(1-sse/variance)


    center_nm = float(origin+center*scale)
    sigma_nm = float(sigma*scale)
    dense_x = np.linspace(x[0], x[-1], 101)
    dense_y = baseline+amplitude*np.exp(-.5*((dense_x-center_nm)/sigma_nm)**2)
    reason = '五点高斯估计·精度未验证'
    return dict(

        fit_available=True,

        reason=reason,
        center_nm=center_nm,
        sigma_nm=sigma_nm,
        baseline_adc=float(baseline),
        amplitude_adc=float(amplitude),
        r_squared=r2,
        rmse_adc=rmse,
        curve_x_nm=dense_x.tolist(),
        curve_adc=dense_y.tolist(),
        optical_accuracy_verified=False,
    )


def fit_temporary_frame(rows, values, warmed_up=True):
    if (len(rows) != len(values) or len(rows) not in range(5, 46, 5)):
        raise ValueError('实时拟合需要1～9个完整五点峰')
    # warmed_up is retained only for old callers; it is not a peak-quality gate.
    return [fit_five_points([r['measured_nm'] for r in rows[i:i+5]], values[i:i+5])
            for i in range(0, len(rows), 5)]
