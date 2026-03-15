"""
Trade Viewer — candlestick plot module.
Works with base (threshold_pct), ATR, and Bollinger Band entry strategies.

Usage in notebook:
    from odbicie.plot import show_trade_viewer
    show_trade_viewer(trades_df, dfs_1d, tpm, slm, ttpm, mhb)
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import ipywidgets as widgets
from IPython.display import display, clear_output


def _val(p):
    """Extract .value from ipywidgets, or just return the number."""
    return p.value if hasattr(p, 'value') else p


def _simulate_barriers(entry_price, atr, highs, lows, tp_mult, sl_mult, ttp_mult,
                       max_holding_bars=15, active_trailing_sl=False,
                       sl_trail_mult=2.0, max_loss_pct=1.0, time_decay_sl=False,
                       strategy_type='tbm'):
    vol_val = atr if (not np.isnan(atr) and atr != 0) else entry_price * 0.02

    sl_vals, tp_act_vals, ttp_vals = [], [], []

    if strategy_type.lower() == 'simple':
        sl_price = entry_price - (vol_val * sl_mult)
        tp_price = entry_price + (vol_val * tp_mult)
        
        for i, (h, l) in enumerate(zip(highs, lows)):
            sl_vals.append(sl_price)
            tp_act_vals.append(tp_price)
            ttp_vals.append(np.nan)
        return sl_vals, tp_act_vals, ttp_vals, None

    # TBM logic
    calculated_sl     = entry_price - vol_val * sl_mult
    hard_cap_sl       = entry_price * (1 - max_loss_pct)
    current_sl        = max(calculated_sl, hard_cap_sl)
    tp_activation     = entry_price + vol_val * tp_mult
    tp_trail_distance = vol_val * ttp_mult

    is_tp_trailing  = False
    current_tp_stop = np.nan
    activation_bar  = None
    initial_sl      = current_sl
    max_sl_distance = entry_price - current_sl

    for i, (h, l) in enumerate(zip(highs, lows)):
        # 1. Check Trailing Take Profit activation
        if not is_tp_trailing and h >= tp_activation:
            is_tp_trailing = True
            activation_bar = i
            current_tp_stop = tp_activation - tp_trail_distance
            new_tp = h - tp_trail_distance
            if new_tp > current_tp_stop:
                current_tp_stop = new_tp

        # 1.5 Time decay SL
        if time_decay_sl:
            current_sl_distance = max_sl_distance * (1 - (i / max_holding_bars))
            new_decay_sl = entry_price - current_sl_distance
            if new_decay_sl > current_sl:
                current_sl = new_decay_sl

        # 4. Update trailing SL based on high
        if active_trailing_sl:
            new_sl = h - (vol_val * sl_trail_mult)
        else:
            new_sl = h - (vol_val * sl_mult)

        if new_sl > current_sl:
            current_sl = new_sl

        # Update trailing TP (if active)
        if is_tp_trailing:
            new_tp = h - tp_trail_distance
            if new_tp > current_tp_stop:
                current_tp_stop = new_tp

        sl_vals.append(current_sl)
        tp_act_vals.append(tp_activation)
        ttp_vals.append(current_tp_stop)

    return sl_vals, tp_act_vals, ttp_vals, activation_bar


def _build_entry_info(row):
    """
    Build entry-method specific info lines.
    Auto-detects whether the trade came from:
      - odbicie.py        (threshold_pct)
      - odbicie_atr.py   (atr_factor / atr_period)
      - odbicie_bb.py    (bb_period / bb_std)
    """
    lines = []
    if 'entry_atr' in row.index and not pd.isna(row.get('entry_atr')):
        lines.append(('Entry ATR', f"{float(row['entry_atr']):.4f}"))

    if 'atr_factor' in row.index:
        # ATR-based entry
        lines.append(('ATR period',      str(int(row['atr_period']))))
        lines.append(('ATR factor',      f"{float(row['atr_factor']):.1f}"))
        lines.append(('Signal ATR',      f"{float(row['signal_atr']):.4f}"))
        lines.append(('Pullback dist',   f"${float(row['pullback_distance']):.2f}"))
        lines.append(('Pullback %',      f"{float(row['pullback_pct']):.2f}%"))
    elif 'bb_period' in row.index:
        # BB-based entry
        lines.append(('BB period',       str(int(row['bb_period']))))
        lines.append(('BB std mult',     f"{float(row['bb_std']):.1f}"))
        lines.append(('BB lower',        f"${float(row['bb_lower']):.2f}"))
        lines.append(('BB middle',       f"${float(row['bb_middle']):.2f}"))
        lines.append(('BB upper',        f"${float(row['bb_upper']):.2f}"))
        lines.append(('BB bandwidth',    f"{float(row['bb_bandwidth']):.4f}"))
        if 'rsi_at_entry' in row.index and not pd.isna(row.get('rsi_at_entry')):
            lines.append(('RSI at entry',   f"{float(row['rsi_at_entry']):.1f}"))
    elif 'threshold_pct' in row.index:
        # Percentage-based entry
        lines.append(('Threshold',       f"{float(row['threshold_pct'])*100:.1f}%"))
    else:
        lines.append(('Entry method',    'unknown'))

    return lines


def _draw_indicator_overlay(ax, df, row, date_to_x):
    """
    Draw strategy-specific overlays on the candlestick chart:
      - BB strategy : Bollinger Bands (upper, lower as shaded channel; middle dashed)
      - ATR strategy: ATR value plotted as a secondary annotation line at constant level

    Args:
        ax        : The candlestick matplotlib Axes.
        df        : The price DataFrame sliced to the visible window.
        row       : The trade row (pd.Series) from trades_df.
        date_to_x : Mapping from datetime index → integer x position.
    """
    def x_for(ts):
        ts = pd.to_datetime(ts)
        if ts in date_to_x:
            return date_to_x[ts]
        diffs = [(abs((d - ts).total_seconds()), x) for d, x in date_to_x.items()]
        return min(diffs, key=lambda t: t[0])[1]

    if 'bb_period' in row.index:
        # ── Bollinger Bands overlay ──
        period   = int(row['bb_period'])
        std_mult = float(row['bb_std'])
        close    = df['Close']
        middle   = close.rolling(window=period, min_periods=period).mean()
        std      = close.rolling(window=period, min_periods=period).std(ddof=0)
        upper    = middle + std * std_mult
        lower    = middle - std * std_mult

        xs      = [x_for(ts) for ts in df.index]
        mid_arr = middle.values
        upr_arr = upper.values
        lwr_arr = lower.values

        # Filter out NaN bars
        valid = ~(np.isnan(mid_arr) | np.isnan(upr_arr) | np.isnan(lwr_arr))
        xs_v   = [xs[i] for i in range(len(xs)) if valid[i]]
        mid_v  = mid_arr[valid]
        upr_v  = upr_arr[valid]
        lwr_v  = lwr_arr[valid]

        if xs_v:
            ax.plot(xs_v, mid_v, color='#1565c0', lw=1.2, ls='--',
                    alpha=0.75, label=f'BB middle ({period})', zorder=2)
            ax.plot(xs_v, upr_v, color='#5c6bc0', lw=1.0, ls='-',
                    alpha=0.55, label='BB upper', zorder=2)
            ax.plot(xs_v, lwr_v, color='#5c6bc0', lw=1.0, ls='-',
                    alpha=0.55, label='BB lower', zorder=2)
            ax.fill_between(xs_v, lwr_v, upr_v,
                            color='#9fa8da', alpha=0.10, zorder=1)

    elif 'atr_factor' in row.index:
        # ── ATR overlay: plot recomputed rolling ATR as a line on a twin y-axis ──
        period = int(row['atr_period'])
        high, low, prev_close = df['High'], df['Low'], df['Close'].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(window=period, min_periods=period).mean()

        xs      = [x_for(ts) for ts in df.index]
        atr_arr = atr_series.loc[df.index].values

        valid = ~np.isnan(atr_arr)
        xs_v   = [xs[i] for i in range(len(xs)) if valid[i]]
        atr_v  = atr_arr[valid]

        if xs_v:
            ax2 = ax.twinx()
            ax2.plot(xs_v, atr_v, color='#ff6f00', lw=1.4, ls='-.',
                     alpha=0.7, label=f'ATR ({period})')
            ax2.set_ylabel(f'ATR ({period})', fontsize=8, color='#ff6f00')
            ax2.tick_params(axis='y', labelcolor='#ff6f00', labelsize=7)
            ax2.spines['right'].set_edgecolor('#ff6f00')
            # Add to the main legend manually
            ax.plot([], [], color='#ff6f00', lw=1.4, ls='-.',
                    alpha=0.7, label=f'ATR ({period})')


def show_trade_viewer(trades_df, dfs_1d, tpm, slm, ttpm, mhb, exit_reason="all",
                      active_trailing_sl=False, sl_trail_mult=2.0,
                      max_loss_pct=1.0, time_decay_sl=False, exit_on_close=False,
                      strategy_type="tbm"):
    """
    Display an interactive candlestick trade viewer widget.
    Parameters can be plain numbers or ipywidgets.
    `strategy_type` can be "tbm" or "simple".
    """

    tp_mult  = float(tpm)
    sl_mult  = float(slm)
    ttp_mult = float(ttpm)
    max_bars = int(mhb)

    PARAMS_BOX = {
        'Strategy':         strategy_type.upper(),
        'TP mult':          tp_mult,
        'SL mult':          sl_mult,
        'Max holding bars': max_bars,
        'Exit on close':    exit_on_close,
    }
    
    if strategy_type.lower() == 'tbm':
        PARAMS_BOX['TTP trail mult'] = ttp_mult

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
            max_holding_bars=max_bars, active_trailing_sl=active_trailing_sl,
            sl_trail_mult=sl_trail_mult, max_loss_pct=max_loss_pct,
            time_decay_sl=time_decay_sl, strategy_type=strategy_type
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

    # Extract all unique exit reasons for the filter dropdown
    unique_reasons = sorted(list({d['row']['exit_reason'] for d in trades_data if 'exit_reason' in d['row']}))
    reasons_options = ['All'] + unique_reasons

    # ── Widget ──
    state = {'idx': 0, 'filtered_indices': list(range(len(trades_data)))}
    out = widgets.Output()

    def plot_trade():
        if not state['filtered_indices']:
            with out:
                clear_output(wait=True)
                print("Brak transakcji dla wybranego filtru.")
            return
            
        index = state['filtered_indices'][state['idx']]
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

                # Candlesticks
                dates     = df.index
                n         = len(dates)
                date_to_x = {d: i for i, d in enumerate(dates)}

                with plt.ioff():
                    fig = plt.figure(figsize=(18, 8))
                    gs  = GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
                    ax  = fig.add_subplot(gs[0, 0])
                    ax_info = fig.add_subplot(gs[0, 1])

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

                # ── Indicator overlay (BB / ATR) ──
                _draw_indicator_overlay(ax, df, row, date_to_x)

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

                is_tbm = strategy_type.lower() == 'tbm'

                if sl_xs:
                    label = 'Trailing SL' if is_tbm else 'Stop Loss'
                    ax.step(sl_xs, sl_ys, where='post', color='#ef5350', lw=1.4,
                            ls='--', label=label, zorder=4)
                if tp_xs:
                    label = 'TP activation' if is_tbm else 'Take Profit'
                    ax.step(tp_xs, tp_ys, where='post', color='#4caf50', lw=1.4,
                            ls='--', label=label, zorder=4)
                if ttp_xs and is_tbm:
                    ax.step(ttp_xs, ttp_ys, where='post', color='#ff9800', lw=1.8,
                            ls='-', label='Trailing TP stop', zorder=4)

                # TP activation marker (TBM only)
                act_bar = data['activation_bar']
                if act_bar is not None and act_bar < len(future) and is_tbm:
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

                # Params box
                ptxt = '\n'.join(f'{k}: {v}' for k, v in PARAMS_BOX.items())
                ax.text(0.01, 0.99, f'Simulation Parameters\n{ptxt}',
                        transform=ax.transAxes, fontsize=9, va='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round,pad=0.5', fc='#e3f2fd', alpha=0.85, ec='#90caf9'))

                view_idx = state['idx'] + 1
                total_filtered = len(state['filtered_indices'])
                ax.set_title(f'[{symbol}]  Trade {view_idx} / {total_filtered}',
                             fontsize=13, fontweight='bold')
                ax.set_ylabel('Cena ($)')
                ax.grid(True, alpha=0.2)
                ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9, ncol=2)

                # ── Info panel ──
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
                ]

                # Auto-detect entry method and append relevant info
                info.extend(_build_entry_info(row))

                info.extend([
                    ('',             ''),
                    ('Exit time',    str(exit_time.date())),
                    ('Exit price',   f"${exit_price:.2f}"),
                    ('Exit reason',  row['exit_reason']),
                    ('Hold bars',    str(int(row['hold_bars']))),
                    ('',             ''),
                    ('Ideal entry',  f"${opt_price:.2f}"),
                    ('Ideal date',   str(opt_idx.date()) if hasattr(opt_idx, 'date') else str(opt_idx)),
                ])

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
                display(fig)
                plt.close(fig)

            except Exception:
                import traceback
                traceback.print_exc()

    def update_filter(change):
        selected = change['new']
        if selected == 'All':
            state['filtered_indices'] = list(range(len(trades_data)))
        else:
            state['filtered_indices'] = [
                i for i, d in enumerate(trades_data) 
                if d['row'].get('exit_reason') == selected
            ]
        state['idx'] = 0
        plot_trade()

    def on_next(b):
        if not state['filtered_indices']: return
        state['idx'] = (state['idx'] + 1) % len(state['filtered_indices'])
        plot_trade()

    def on_prev(b):
        if not state['filtered_indices']: return
        state['idx'] = (state['idx'] - 1) % len(state['filtered_indices'])
        plot_trade()

    prev_btn = widgets.Button(description='⬅️ Poprzednia', button_style='warning')
    next_btn = widgets.Button(description='Następna ➡️',   button_style='info')
    
    filter_dropdown = widgets.Dropdown(
        options=reasons_options,
        value='All',
        description='Exit Reason:',
        style={'description_width': 'initial'}
    )
    filter_dropdown.observe(update_filter, names='value')

    prev_btn.on_click(on_prev)
    next_btn.on_click(on_next)

    display(widgets.HBox([filter_dropdown, prev_btn, next_btn]))
    display(out)
    plot_trade()
