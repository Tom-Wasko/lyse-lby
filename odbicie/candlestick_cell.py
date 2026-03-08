"""
Candlestick Trade Viewer — importable module.
Usage in notebook:
    from candlestick_cell import show_trade_viewer
    show_trade_viewer(trades_df, dfs_1d, tpm, slm, ttpm, mhb)
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import ipywidgets as widgets
from IPython.display import display, clear_output


def _val(p):
    """Extract .value from ipywidgets, or just return the number."""
    return p.value if hasattr(p, 'value') else p


def _simulate_barriers(entry_price, atr, highs, lows, tp_mult, sl_mult, ttp_mult):
    vol_val = atr if (not np.isnan(atr) and atr != 0) else entry_price * 0.02

    sl_vals, tp_act_vals, ttp_vals = [], [], []

    current_sl        = entry_price - vol_val * sl_mult
    tp_activation     = entry_price + vol_val * tp_mult
    tp_trail_distance = vol_val * ttp_mult

    is_tp_trailing  = False
    current_tp_stop = np.nan
    activation_bar  = None

    for i, (h, l) in enumerate(zip(highs, lows)):
        if not is_tp_trailing and h >= tp_activation:
            is_tp_trailing = True
            activation_bar = i
            current_tp_stop = tp_activation - tp_trail_distance
            new_tp = h - tp_trail_distance
            if new_tp > current_tp_stop:
                current_tp_stop = new_tp

        new_sl = h - vol_val * sl_mult
        if new_sl > current_sl:
            current_sl = new_sl

        if is_tp_trailing:
            new_tp = h - tp_trail_distance
            if new_tp > current_tp_stop:
                current_tp_stop = new_tp

        sl_vals.append(current_sl)
        tp_act_vals.append(tp_activation)
        ttp_vals.append(current_tp_stop)

    return sl_vals, tp_act_vals, ttp_vals, activation_bar


def show_trade_viewer(trades_df, dfs_1d, tpm, slm, ttpm, mhb, exit_reason="all"):
    """
    Display an interactive candlestick trade viewer widget.
    Parameters can be plain numbers or ipywidgets.
    """
    # We no longer force 'Agg', let the native Jupyter backend handle it.

    tp_mult  = float(tpm)
    sl_mult  = float(slm)
    ttp_mult = float(ttpm)
    max_bars = int(mhb)

    TBM_PARAMS = {
        'TP mult':          tp_mult,
        'SL mult':          sl_mult,
        'TTP trail mult':   ttp_mult,
        'Max holding bars': max_bars,
    }

    # ── Pre-compute trades ──
    trades_data = []

    for _, row in trades_df.iterrows():
        if exit_reason is not None and str(exit_reason).lower() != "all":
            if isinstance(exit_reason, (list, tuple, set)):
                if row['exit_reason'] not in exit_reason:
                    continue
            else:
                if row['exit_reason'] != exit_reason:
                    continue

        symbol      = row['symbol']
        signal_time = pd.to_datetime(row['signal_time'])
        entry_time  = pd.to_datetime(row['entry_time'])
        exit_time   = pd.to_datetime(row['exit_time'])

        if symbol not in dfs_1d:
            continue

        df_full = dfs_1d[symbol]
        start_date = signal_time - pd.Timedelta(days=30)
        end_date   = exit_time   + pd.Timedelta(days=10)
        df = df_full.loc[(df_full.index >= start_date) & (df_full.index <= end_date)].copy()

        if df.empty or not {'Open','High','Low','Close'}.issubset(df.columns):
            continue

        trade_window = df.loc[(df.index >= entry_time) & (df.index <= exit_time)]
        if trade_window.empty:
            continue

        opt_idx   = trade_window['Low'].idxmin()
        opt_price = trade_window.loc[opt_idx, 'Low']

        entry_atr = row.get('entry_atr', np.nan)
        hold = int(row['hold_bars']) if row['hold_bars'] > 0 else max_bars
        future = df.loc[df.index > entry_time].head(hold)

        sl_vals, tp_act_vals, ttp_vals, activation_bar = _simulate_barriers(
            float(row['entry_price']), float(entry_atr) if not pd.isna(entry_atr) else np.nan,
            future['High'].values, future['Low'].values,
            tp_mult, sl_mult, ttp_mult,
        )

        trades_data.append({
            'df': df, 'future': future, 'symbol': symbol, 'row': row,
            'optimal_idx': opt_idx, 'optimal_price': opt_price,
            'sl_vals': sl_vals, 'tp_act_vals': tp_act_vals, 'ttp_vals': ttp_vals,
            'activation_bar': activation_bar,
        })

    if not trades_data:
        print("Brak transakcji do wyświetlenia.")
        return

    # ── Widget ──
    state = {'idx': 0}
    out = widgets.Output()

    def plot_trade(index):
        with out:
            clear_output(wait=True)
            try:
                data   = trades_data[index]
                df     = data['df']
                future = data['future']
                row    = data['row']
                symbol = data['symbol']

                signal_time  = pd.to_datetime(row['signal_time'])
                entry_time   = pd.to_datetime(row['entry_time'])
                exit_time    = pd.to_datetime(row['exit_time'])
                signal_close = float(row['signal_close'])
                entry_price  = float(row['entry_price'])
                exit_price   = float(row['exit_price'])
                opt_idx      = data['optimal_idx']
                opt_price    = data['optimal_price']

                fig = plt.figure(figsize=(18, 8))
                gs  = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
                ax  = fig.add_subplot(gs[0, 0])
                ax_info = fig.add_subplot(gs[0, 1])

                # Candlesticks
                dates     = df.index
                n         = len(dates)
                date_to_x = {d: i for i, d in enumerate(dates)}

                for i, (dt, r) in enumerate(df.iterrows()):
                    o, h, l, c = r['Open'], r['High'], r['Low'], r['Close']
                    color = '#26a69a' if c >= o else '#ef5350'
                    body_h = abs(c - o) or (h - l) * 0.01
                    ax.bar(i, body_h, bottom=min(o, c), width=0.6,
                           color=color, edgecolor=color, linewidth=0.5)
                    ax.plot([i, i], [l, h], color=color, linewidth=0.8)

                def x_for(ts):
                    ts = pd.to_datetime(ts)
                    if ts in date_to_x:
                        return date_to_x[ts]
                    diffs = [(abs((d - ts).total_seconds()), x) for d, x in date_to_x.items()]
                    return min(diffs, key=lambda t: t[0])[1]

                # SL / TP / TTP lines
                sl_xs, sl_ys = [], []
                tp_xs, tp_ys = [], []
                ttp_xs, ttp_ys = [], []
                for bar_i, bar_dt in enumerate(future.index):
                    if bar_i >= len(data['sl_vals']):
                        break
                    xi = x_for(bar_dt)
                    sl_xs.append(xi);  sl_ys.append(data['sl_vals'][bar_i])
                    tp_xs.append(xi);  tp_ys.append(data['tp_act_vals'][bar_i])
                    v = data['ttp_vals'][bar_i]
                    if not np.isnan(v):
                        ttp_xs.append(xi); ttp_ys.append(v)

                if sl_xs:
                    ax.step(sl_xs, sl_ys, where='post', color='#ef5350', lw=1.4,
                            ls='--', label='Trailing SL', zorder=4)
                if tp_xs:
                    ax.step(tp_xs, tp_ys, where='post', color='#4caf50', lw=1.4,
                            ls='--', label='TP activation', zorder=4)
                if ttp_xs:
                    ax.step(ttp_xs, ttp_ys, where='post', color='#ff9800', lw=1.8,
                            ls='-', label='Trailing TP stop', zorder=4)

                # TP activation marker
                act_bar = data['activation_bar']
                if act_bar is not None and act_bar < len(future):
                    act_x = x_for(future.index[act_bar])
                    act_y = data['tp_act_vals'][act_bar]
                    ax.scatter(act_x, act_y, color='#ff9800', s=250, marker='P',
                               edgecolors='black', linewidths=0.8, zorder=7,
                               label='TP → TTP aktivace')

                # Trade markers
                ax.scatter(x_for(signal_time), signal_close, color='darkorange',
                           s=180, marker='D', label='Mackowy sygnał', zorder=6)
                ax.scatter(x_for(opt_idx), opt_price, color='gold',
                           s=300, marker='*', label='Idealny punkt wejścia', zorder=6)
                ax.scatter(x_for(entry_time), entry_price, color='#00e676',
                           s=200, marker='^', label='Faktyczne wejście', zorder=6,
                           edgecolors='black', linewidths=0.5)
                ax.scatter(x_for(exit_time), exit_price, color='#ff1744',
                           s=200, marker='v', label='Wyjście', zorder=6,
                           edgecolors='black', linewidths=0.5)

                for ts in [signal_time, entry_time, exit_time]:
                    ax.axvline(x=x_for(ts), color='gray', ls='--', alpha=0.2)

                # X labels
                step = max(1, n // 12)
                ticks = list(range(0, n, step))
                ax.set_xticks(ticks)
                ax.set_xticklabels([dates[i].strftime('%Y-%m-%d') for i in ticks],
                                   rotation=45, ha='right', fontsize=8)

                # TBM box
                ptxt = '\n'.join(f'{k}: {v}' for k, v in TBM_PARAMS.items())
                ax.text(0.01, 0.99, f'TBM Parameters\n{ptxt}',
                        transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', fc='#e3f2fd', alpha=0.85, ec='#90caf9'))

                ax.set_title(f'[{symbol}]  Trade {index+1} / {len(trades_data)}',
                             fontsize=13, fontweight='bold')
                ax.set_ylabel('Cena ($)')
                ax.grid(True, alpha=0.2)
                ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9, ncol=2)

                # Info panel
                ax_info.axis('off')
                ret = float(row['return_pct'])
                rc  = '#26a69a' if ret > 0 else '#ef5350'

                info = [
                    ('Symbol',       symbol),
                    ('Pattern',      row['pattern']),
                    ('',             ''),
                    ('Signal time',  str(signal_time.date())),
                    ('Signal close', f"${signal_close:.2f}"),
                    ('',             ''),
                    ('Entry time',   str(entry_time.date())),
                    ('Entry price',  f"${entry_price:.2f}"),
                    ('Entry ATR',    f"{float(row['entry_atr']):.4f}"),
                    ('Threshold',    f"{float(row['threshold_pct'])*100:.1f}%"),
                    ('',             ''),
                    ('Exit time',    str(exit_time.date())),
                    ('Exit price',   f"${exit_price:.2f}"),
                    ('Exit reason',  row['exit_reason']),
                    ('Hold bars',    str(int(row['hold_bars']))),
                    ('',             ''),
                    ('Ideal entry',  f"${opt_price:.2f}"),
                    ('Ideal date',   str(opt_idx.date()) if hasattr(opt_idx, 'date') else str(opt_idx)),
                ]

                ax_info.text(0.05, 0.97, 'Trade Information', fontsize=12, fontweight='bold',
                             transform=ax_info.transAxes, va='top')
                y, lh = 0.97 - 0.055, 0.047
                for label, value in info:
                    if label == '':
                        y -= lh * 0.4; continue
                    ax_info.text(0.05, y, f'{label}:', fontsize=9, fontweight='bold',
                                 color='#555', transform=ax_info.transAxes, va='top')
                    ax_info.text(0.55, y, str(value), fontsize=9,
                                 transform=ax_info.transAxes, va='top')
                    y -= lh

                y -= lh * 0.3
                ax_info.text(0.05, y, 'Return:', fontsize=11, fontweight='bold',
                             color='#333', transform=ax_info.transAxes, va='top')
                ax_info.text(0.55, y, f'{ret:+.2f}%', fontsize=14, fontweight='bold',
                             color=rc, transform=ax_info.transAxes, va='top')

                plt.tight_layout()
                plt.show()

            except Exception:
                import traceback
                traceback.print_exc()

    def on_next(b):
        state['idx'] = (state['idx'] + 1) % len(trades_data)
        plot_trade(state['idx'])

    def on_prev(b):
        state['idx'] = (state['idx'] - 1) % len(trades_data)
        plot_trade(state['idx'])

    prev_btn = widgets.Button(description='⬅️ Poprzednia', button_style='warning')
    next_btn = widgets.Button(description='Następna ➡️',   button_style='info')
    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)

    display(widgets.HBox([prev_btn, next_btn]))
    display(out)
    plot_trade(state['idx'])
