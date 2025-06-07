import pymssql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

# 連接 SQL Server 資料庫
conn = pymssql.connect(
    host="127.0.0.1",
    user="MSI/user2",
    password="ezshun0719",
    database="ncu_database",
    charset="utf8"
)
cursor = conn.cursor()

# 使用者輸入股票代碼
stock_code = input("請輸入股票代碼: ").strip()

# 查詢資料（包含MA20、KD值和其他繪圖所需欄位）
cursor.execute("""
    SELECT [Date], [Open], [Close], [High], [Low], [Volume], [MA20], [K_Value], [D_Value]
    FROM StockTrading_TA
    WHERE StockCode = %s
    ORDER BY [Date] ASC
""", (stock_code,))

column_names = [column[0] for column in cursor.description]
rows = [tuple(row) for row in cursor.fetchall()]

# 轉換 DataFrame
df = pd.DataFrame(rows, columns=column_names)

# 將欄位名稱標準化
df.rename(columns={"Date": "date", "Close": "close", "Volume": "volume", "Open": "open", "High": "high", "Low": "low", "K_Value": "K", "D_Value": "D"}, inplace=True)
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# 計算 RSI 和 5日均量（MA20直接從資料表取得）
def add_rsi_and_volume(df, rsi_period=14, ma_volume_period=5):
    # 計算RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 計算5日均量
    df['MA_volume'] = df['volume'].rolling(window=ma_volume_period).mean()
    
    return df

df = add_rsi_and_volume(df)

# 計算 Fibonacci 回撤區間
high = df['close'].max()
low = df['close'].min()
diff = high - low
retracements = {
    "23.6%": high - 0.236 * diff,
    "38.2%": high - 0.382 * diff,
    "50.0%": high - 0.500 * diff,
    "61.8%": high - 0.618 * diff,
    "78.6%": high - 0.786 * diff,
}

print(f"\n=== {stock_code} Fibonacci 回撤位準 ===")
print(f"期間最高價: {high:.2f}")
print(f"期間最低價: {low:.2f}")
for level, price in retracements.items():
    print(f"{level} 回撤: {price:.2f}")

# 取高點附近的 retracement 作為賣出參考位
retracement_high = retracements["23.6%"]
# 依優先順序排列 retracement 作為買進參考位
retracement_lows = [retracements["78.6%"], retracements["61.8%"], retracements["50.0%"]]

# 原本的 Fibonacci + RSI 策略
def detect_fibonacci_signals(df, tolerance=0.015):
    """原本的Fibonacci回撤策略"""
    buy_signals, sell_signals = [], []

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']
        rsi = row['RSI']
        volume = row['volume']
        ma_vol = row['MA_volume']

        volume_spike = volume > ma_vol if pd.notna(ma_vol) else False

        # 買進訊號（按 retracement 優先順序，只取第一個符合者）
        if pd.notna(rsi) and rsi < 44 and volume_spike:
            for rl in retracement_lows:
                if abs(price - rl) / rl <= tolerance:
                    buy_signals.append((row.name, price, rsi, volume, 'Fibonacci'))
                    break

        # 賣出訊號（使用 23.6% 回撤作為參考）
        if pd.notna(rsi) and rsi > 65 and volume_spike and abs(price - retracement_high) / retracement_high <= tolerance:
            sell_signals.append((row.name, price, rsi, volume, 'Fibonacci'))

    return buy_signals, sell_signals

# Granville 簡化邏輯檢測函數
def detect_granville_signals(df, window_size=6, max_break_days=3):
    """
    檢測Granville Rule 2 (假跌破) 和 Rule 6 (假突破) 信號
    """
    granville_buy_signals = []
    granville_sell_signals = []
    
    print(f"\n=== Granville 邏輯檢測 (窗口: {window_size}天, 最大跌破次數: {max_break_days}) ===")
    
    for i in range(window_size, len(df)):
        window = df.iloc[i - window_size + 1:i + 1]  # 取window_size天窗口
        if len(window) < window_size or window['MA20'].isna().any():
            continue
            
        first_day = window.iloc[0]
        last_day = window.iloc[-1]
        middle_days = window.iloc[1:-1]  # 中間天數
        
        # 檢查MA20趨勢方向
        ma20_rising = last_day['MA20'] > first_day['MA20']
        ma20_falling = last_day['MA20'] < first_day['MA20']
        
        # Rule 2: 假跌破買入邏輯
        if ma20_rising:
            # 條件1: 第一天股價在MA20之上
            first_above_ma = first_day['close'] >= first_day['MA20']
            
            # 條件2: 期間內跌破MA20次數不超過max_break_days次
            break_count = sum(1 for _, day in middle_days.iterrows() if day['close'] < day['MA20'])
            break_days_ok = break_count <= max_break_days
            
            # 條件3: 最後一天重新站上MA20
            last_above_ma = last_day['close'] >= last_day['MA20']
            
            # 額外條件: RSI和成交量
            rsi_ok = last_day['RSI'] < 55 if pd.notna(last_day['RSI']) else True
            volume_ok = last_day['volume'] > last_day['MA_volume'] * 1.1 if pd.notna(last_day['MA_volume']) else True
            
            if first_above_ma and break_days_ok and last_above_ma and rsi_ok and volume_ok:
                granville_buy_signals.append((
                    last_day.name, 
                    last_day['close'], 
                    last_day['RSI'], 
                    last_day['volume'],
                    'Granville_Rule2'
                ))
                print(f"📈 Granville Rule2 買入: {last_day.name.date()} | 價格: {last_day['close']:.2f} | 跌破次數: {break_count}")
        
        # Rule 6: 假突破賣出邏輯
        if ma20_falling:
            # 條件1: 第一天股價在MA20之下
            first_below_ma = first_day['close'] <= first_day['MA20']
            
            # 條件2: 期間內站上MA20次數不超過max_break_days次
            break_count = sum(1 for _, day in middle_days.iterrows() if day['close'] > day['MA20'])
            break_days_ok = break_count <= max_break_days
            
            # 條件3: 最後一天再次跌回MA20之下
            last_below_ma = last_day['close'] <= last_day['MA20']
            
            # 額外條件: RSI和成交量
            rsi_ok = last_day['RSI'] > 50 if pd.notna(last_day['RSI']) else True
            volume_ok = last_day['volume'] > last_day['MA_volume'] * 1.1 if pd.notna(last_day['MA_volume']) else True
            
            if first_below_ma and break_days_ok and last_below_ma and rsi_ok and volume_ok:
                granville_sell_signals.append((
                    last_day.name, 
                    last_day['close'], 
                    last_day['RSI'], 
                    last_day['volume'],
                    'Granville_Rule6'
                ))
                print(f"📉 Granville Rule6 賣出: {last_day.name.date()} | 價格: {last_day['close']:.2f} | 突破次數: {break_count}")
    
    return granville_buy_signals, granville_sell_signals

# 執行兩種策略
print(f"\n=== 執行 Fibonacci 策略 ===")
fib_buy_signals, fib_sell_signals = detect_fibonacci_signals(df, tolerance=0.015)

print(f"\n=== 執行 Granville 策略 ===")
granville_buy_signals, granville_sell_signals = detect_granville_signals(df, window_size=6, max_break_days=3)

# 合併所有信號
all_buy_signals = fib_buy_signals + granville_buy_signals
all_sell_signals = fib_sell_signals + granville_sell_signals

# 去除重複信號（同一天只保留一個）
buy_signals_dict = {}
for signal in all_buy_signals:
    date = signal[0]
    if date not in buy_signals_dict:
        buy_signals_dict[date] = signal

sell_signals_dict = {}
for signal in all_sell_signals:
    date = signal[0]
    if date not in sell_signals_dict:
        sell_signals_dict[date] = signal

final_buy_signals = list(buy_signals_dict.values())
final_sell_signals = list(sell_signals_dict.values())

# 按日期排序
final_buy_signals.sort(key=lambda x: x[0])
final_sell_signals.sort(key=lambda x: x[0])

# 印出最終結果
print(f"\n" + "="*80)
print(f"=== 最終買賣信號結果 ===")
print(f"="*80)

print(f"\n【買進訊號】(總共 {len(final_buy_signals)} 個)")
print("-" * 85)
print(f"{'日期':<12} {'價格':<8} {'RSI':<6} {'成交量':<12} {'策略來源':<20}")
print("-" * 85)
for date, price, rsi, volume, strategy in final_buy_signals:
    rsi_str = f"{rsi:.1f}" if pd.notna(rsi) else "N/A"
    print(f"{date.date()} {price:>7.2f} {rsi_str:>5} {volume:>11.0f} {strategy}")

print(f"\n【賣出訊號】(總共 {len(final_sell_signals)} 個)")
print("-" * 85)
print(f"{'日期':<12} {'價格':<8} {'RSI':<6} {'成交量':<12} {'策略來源':<20}")
print("-" * 85)
for date, price, rsi, volume, strategy in final_sell_signals:
    rsi_str = f"{rsi:.1f}" if pd.notna(rsi) else "N/A"
    print(f"{date.date()} {price:>7.2f} {rsi_str:>5} {volume:>11.0f} {strategy}")

# 統計各策略貢獻
print(f"\n【策略貢獻統計】")
fibonacci_buy_count = len([s for s in final_buy_signals if s[4] == 'Fibonacci'])
granville_buy_count = len([s for s in final_buy_signals if 'Granville' in s[4]])
fibonacci_sell_count = len([s for s in final_sell_signals if s[4] == 'Fibonacci'])
granville_sell_count = len([s for s in final_sell_signals if 'Granville' in s[4]])

print(f"Fibonacci策略: 買入 {fibonacci_buy_count} 個, 賣出 {fibonacci_sell_count} 個")
print(f"Granville策略: 買入 {granville_buy_count} 個, 賣出 {granville_sell_count} 個")
print(f"總計: 買入 {len(final_buy_signals)} 個, 賣出 {len(final_sell_signals)} 個")

# 如果有重疊信號，顯示統計
total_before_dedup = len(all_buy_signals) + len(all_sell_signals)
total_after_dedup = len(final_buy_signals) + len(final_sell_signals)
if total_before_dedup > total_after_dedup:
    print(f"去重前信號總數: {total_before_dedup}, 去重後: {total_after_dedup}")

# 畫圖
print(f"\n=== 開始繪製技術分析圖表 ===")

# 檢查KD資料
has_kd = 'K' in df.columns and 'D' in df.columns and df['K'].notna().sum() > 0 and df['D'].notna().sum() > 0

if has_kd:
    print("✅ 檢測到KD資料，使用雙面板圖表")
    print(f"K值範圍: {df['K'].min():.2f} - {df['K'].max():.2f}")
    print(f"D值範圍: {df['D'].min():.2f} - {df['D'].max():.2f}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1], sharex=True)
else:
    print("❌ 無KD資料，使用單面板圖表")
    fig, ax1 = plt.subplots(figsize=(16, 8))
    ax2 = None

candlestick_width = 0.6

# 上圖：主要價格圖表
for date, row in df.iterrows():
    color = 'red' if row['close'] > row['open'] else 'green'
    ax1.add_patch(Rectangle(
        (mdates.date2num(date) - candlestick_width / 2, min(row['open'], row['close'])),
        candlestick_width,
        abs(row['close'] - row['open']),
        color=color,
        zorder=2
    ))
    ax1.plot([date, date], [row['low'], row['high']], color=color, linewidth=1, zorder=1)

# MA20
ax1.plot(df.index, df['MA20'], label='MA20', color='orange', linewidth=2)

# 全域Fibonacci回撤線
for label, level in retracements.items():
    ax1.hlines(y=level, xmin=df.index[0], xmax=df.index[-1], 
              colors='purple', linestyles='-', linewidth=1.5, alpha=0.8, 
              label=f'Fib {label}')

# 標記買賣信號
buy_dates = [signal[0] for signal in final_buy_signals]
sell_dates = [signal[0] for signal in final_sell_signals]

if buy_dates:
    # 買入信號：放在K棒下方
    buy_display_prices = []
    for date in buy_dates:
        if date in df.index:
            low_price = df.loc[date, 'low']
            offset = (df['high'].max() - df['low'].min()) * 0.02
            buy_display_prices.append(low_price - offset)
    
    ax1.scatter(buy_dates, buy_display_prices, marker='^', color='blue', edgecolors='white',
               s=80, linewidths=1, label='Buy Signal', zorder=5)

if sell_dates:
    # 賣出信號：放在K棒上方
    sell_display_prices = []
    for date in sell_dates:
        if date in df.index:
            high_price = df.loc[date, 'high']
            offset = (df['high'].max() - df['low'].min()) * 0.02
            sell_display_prices.append(high_price + offset)
    
    ax1.scatter(sell_dates, sell_display_prices, marker='v', color='red', edgecolors='white',
               s=80, linewidths=1, label='Sell Signal', zorder=5)

# 格式化上圖
ax1.set_title(f"{stock_code} Fibonacci + Granville", fontsize=14)
ax1.set_ylabel("Price")
ax1.grid(True, alpha=0.3)

# 圖例
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

# 下圖：KD指標
if has_kd and ax2 is not None:
    print("📊 繪製KD指標圖...")
    
    # 繪製K線和D線
    ax2.plot(df.index, df['K'], label='K', color='blue', linewidth=1.5)
    ax2.plot(df.index, df['D'], label='D', color='red', linewidth=1.5)
    
    # KD參考線
    ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.7, label='Overbought(80)')
    ax2.axhline(y=20, color='gray', linestyle='--', alpha=0.7, label='Oversold(20)')
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    
    # 在KD圖上標記買賣信號點
    if buy_dates:
        valid_buy_dates = []
        buy_k_values = []
        for date in buy_dates:
            if date in df.index and pd.notna(df.loc[date, 'K']):
                valid_buy_dates.append(date)
                buy_k_values.append(df.loc[date, 'K'])
        
        if valid_buy_dates:
            ax2.scatter(valid_buy_dates, buy_k_values, marker='o', color='blue', 
                       s=60, edgecolors='white', linewidths=1, zorder=5, alpha=0.8)
            print(f"標記了 {len(valid_buy_dates)} 個買入信號點")
    
    if sell_dates:
        valid_sell_dates = []
        sell_k_values = []
        for date in sell_dates:
            if date in df.index and pd.notna(df.loc[date, 'K']):
                valid_sell_dates.append(date)
                sell_k_values.append(df.loc[date, 'K'])
        
        if valid_sell_dates:
            ax2.scatter(valid_sell_dates, sell_k_values, marker='o', color='red', 
                       s=60, edgecolors='white', linewidths=1, zorder=5, alpha=0.8)
            print(f"標記了 {len(valid_sell_dates)} 個賣出信號點")
    
    ax2.set_ylabel("KD Value")
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_title("KD Stochastic Oscillator", fontsize=12)

# X軸格式化
if has_kd and ax2 is not None:
    # 雙面板：只在下圖顯示X軸標籤
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.set_xticklabels([])  # 隱藏上圖的X軸標籤
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
else:
    # 單面板：在主圖顯示X軸標籤
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.show()

# 關閉資料庫連接
cursor.close()
conn.close()

print(f"\n程式執行完成！")
