# High-Performance Tick Data Pipeline & Multi-Tick Strategy Backtester

데이트레이딩용 고성능 틱데이터(Tick Data) 처리, DuckDB 기반 로컬 DB 저장, N-틱 OHLCV 동적 리샘플링, 3개 카테고리(추세, 모멘텀, 변동성) 30대 핵심 지표 및 백테스트, Plotly 대화형 차트 및 ReportLab PDF 자동 리포트 생성 통합 퀀트 플랫폼입니다.

---

## 🌟 Key Features

1. **Multi-Year Single Unified Continuous Chart Data & Ingestion**:
   - `csv/` 폴더 내의 다년간, 수천 개 이상의 일별 CSV 틱데이터 파일(예: `170328.csv` -> 2017년 3월 28일, `20170328.csv`, `230410.csv` 등)을 자동 감지하여 하나의 연속된 단일 시계열(Single Continuous Series)로 DuckDB에 완벽 통합.
   - 시간만 있는 컬럼(`15:45:00` 등)과 파일명 날짜를 자동 결합하여 밀리초 수준의 완전한 타임스탬프 생성.
   - **자동 증분 업데이트(Incremental Sync)**: 신규 일별 CSV 파일 추가 시 중복 없이 신규 파일만 즉시 감지하여 추가 적재.
   - CP949, EUC-KR, UTF-8, UTF-8-SIG 등 다양한 증권사 인코딩 자동 처리.

2. **Period-Based Backtesting (기간별 백테스트 기능)**:
   - **전체 기간(All)**, **연도별(Yearly, e.g. 2017, 2018, 2024...)**, **월별(Monthly, e.g. 2017-03...)**, **사용자 지정 기간(Custom Date Range)** 선택 백테스트 지원.
   - 백테스트 결과 하단에 **연도별/월별 성과 분할 요약표(Period Breakdown Matrix)** 실시간 렌더링.

3. **Dynamic Tick Resampling**:
   - 단일 틱 단위(`N-Tick`) 및 다중 주기 스위핑(`Start=1000, End=5000, Step=1000`) 지원.
   - 정확한 OHLCV + VWAP + Volume + Tick Count 산출.

4. **30 Core Indicators (Vectorized NumPy/Pandas)**:
   - C-바인딩 이슈 없이 순수 파이썬/넘파이로 구현된 초고속 30대 기술적 지표.
   - **추세(Trend, 10개)**: EMA Cross, MACD, Ichimoku, Supertrend, ADX+DI, Donchian Channel, VWAP, Parabolic SAR, Keltner Channel, TRIX
   - **모멘텀(Momentum, 10개)**: RSI, Stochastic, CCI, ROC, MFI, Williams %R, CMO, Stochastic RSI, Ultimate Oscillator, TSI
   - **변동성(Volatility, 10개)**: Bollinger Bands, ATR Breakout, Keltner Squeeze, Chaikin Volatility, Historical Volatility, Ulcer Index, StdDev Band, Choppiness Index, Mass Index, RVI

5. **Glassmorphism Dark Web GUI Dashboard**:
   - 로컬 웹서버 기반의 반투명 글래스모피즘 인터랙티브 UI.
   - 기간 필터 탭, 대상 폴더 지정 폼, 전략 랭킹 매트릭스 및 연도별/월별 성과 비교표 제공.

---

## 📁 Directory Structure

```
d:/coding/future_kr_csv/
├── config.py                   # 수수료, 슬리피지, 기본 경로 등 설정
├── generate_sample_data.py     # 모의 일별 CSV 틱데이터 생성기
├── main.py                     # 전체 파이프라인 원클릭 실행 스크립트
├── requirements.txt            # 의존성 패키지 목록
├── data/
│   ├── raw_csv/                # 일별 틱데이터 CSV 저장 폴더
│   └── market_data.duckdb      # DuckDB 로컬 데이터베이스 파일
├── engine/
│   ├── data_engine.py          # CSV 적재 및 증분 업데이트 엔진
│   ├── resampler.py            # N-Tick OHLCV 바 생성기
│   ├── indicators.py           # 30개 지표 (Trend 10, Momentum 10, Volatility 10)
│   ├── strategy.py             # 시그널 생성기 및 카테고리 앙상블
│   ├── backtester.py           # 벡터화 백테스트 엔진
│   └── reporter.py             # Plotly 차트 및 ReportLab PDF 생성기
└── output/
    ├── charts/                 # 생성된 HTML 인터랙티브 차트
    └── reports/                # 생성된 PDF 종합 분석 리포트
```

---

## 🚀 Quick Start

### 1. 독립 실행 파일 (`QuantTerminal.exe`) 실행 (추천)
- **더블클릭 실행**: 브라우저에 글래스모피즘 다크 대시보드가 열리며, 원하는 대상 폴더(`csv` 또는 사용자 임의 경로) 및 기간(전체/연도별/월별/직접지정)을 선택하여 백테스트를 실행할 수 있습니다.
- **콘솔 CLI 모드 실행**:
  ```bash
  # 기본 csv 폴더 전체 기간 실행
  .\dist\QuantTerminal.exe --cli

  # 특정 연도(2017년) 지정 백테스트
  .\dist\QuantTerminal.exe --cli --csv-folder="csv" --year=2017

  # 특정 월(2024년 10월) 지정 백테스트
  .\dist\QuantTerminal.exe --cli --csv-folder="csv" --month=2024-10

  # 사용자 지정 날짜 범위 백테스트
  .\dist\QuantTerminal.exe --cli --csv-folder="csv" --start-date="2017-03-28" --end-date="2017-03-30"
  ```

### 2. 파이썬 소스코드 직접 실행
```bash
# GUI 대시보드 실행
python main.py

# 연도별 CLI 실행
python main.py --cli --csv-folder="csv" --year=2017
```

### 3. 원클릭 실행 파일 재빌드
```bash
python build_exe.py
# 또는
build.bat
```
