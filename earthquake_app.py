import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pydeck as pdk

# 1. 페이지 레이아웃 설정
st.set_page_config(
    page_title="지진 위험도 시각화 대시보드",
    page_icon="🌋",
    layout="wide"
)

st.title("🌋 지진 데이터 분석 및 위험도 지도 배포 시스템")
st.markdown("`earthquake.csv`에서 500개의 데이터를 무작위 샘플링하고, AI 모델을 활용해 지진 위험도를 지도 위에 실시간 시각화합니다.")

# 2. 데이터 자동 샘플링 함수 (캐싱 적용)
@st.cache_data
def load_and_sample_data(file_path, n_samples=500):
    try:
        df = pd.read_csv(file_path)
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"CSV 파일을 가져오는 중 오류가 발생했습니다: {e}")
        return None

# 3. 모델 파일 로드 함수 (안전장치 유지)
def load_model_safely(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        # 가상 모델 정의: 숫자로 변환된 데이터를 받아 안전하게 숫자로 반환하도록 설계
        class DummyModel:
            def predict(self, X):
                try:
                    return X.iloc[:, 0].astype(float) * 0.1 + X.iloc[:, 2].astype(float) * 0.9
                except:
                    return np.zeros(len(X))
        st.error(f"⚠️ 모델 파일(.pkl)을 로드하지 못했습니다 ({e}). 대신 가상 예측 모델 모드로 대시보드를 구동합니다.")
        return DummyModel()

# 파일 로드 실행
df = load_and_sample_data('earthquake.csv', n_samples=500)
model = load_model_safely('earthquake_model.pkl')

if df is not None:
    # 📌 [실제 확인된 CSV 컬럼명 맞춤 설정]
    actual_cols = {
        '위도': 'latitude',
        '경도': 'longitude',
        '규모': 'magnitudo',  
        '진원깊이': 'depth',   
        '영향도': 'significance' 
    }

    # 수치형 데이터 컬럼들을 강제로 숫자 타입으로 변환 및 결측치 방어
    for col_key, real_col_name in actual_cols.items():
        if real_col_name in df.columns:
            df[real_col_name] = pd.to_numeric(df[real_col_name], errors='coerce')
            # 전체가 NaN일 경우를 대비해 기본값 0.0 방어 후 평균값 채우기
            if df[real_col_name].isna().all():
                df[real_col_name] = df[real_col_name].fillna(0.0)
            else:
                df[real_col_name] = df[real_col_name].fillna(df[real_col_name].mean())
        else:
            df[real_col_name] = 0.0

    st.success(f"데이터 로드 및 수치 변환 성공! (컬럼 매칭 완료 -> 규모: magnitudo / 깊이: depth)")

    # 4. 모델 예측 진행
    try:
        features = df[[actual_cols['진원깊이'], actual_cols['영향도'], actual_cols['규모']]]
        predictions = model.predict(features)
        
        if hasattr(predictions, "flatten"):
            df['risk_score'] = pd.to_numeric(predictions.flatten(), errors='coerce')
        else:
            df['risk_score'] = pd.to_numeric(predictions, errors='coerce')
            
    except Exception as e:
        df['risk_score'] = df[actual_cols['규모']]

    # 결측치 및 정규화 최종 방어
    df['risk_score'] = df['risk_score'].fillna(0.0).astype(float)
    min_score = float(df['risk_score'].min())
    max_score = float(df['risk_score'].max())
    
    if max_score - min_score > 0:
        df['normalized_risk'] = (df['risk_score'] - min_score) / (max_score - min_score + 1e-5)
    else:
        df['normalized_risk'] = 0.5

    # 5. 사이드바 조작 필터 기능 (🚨 웹 서버 환경 에러 집중 방어 구간)
    st.sidebar.header("🔍 데이터 필터")
    
    min_mag = float(df[actual_cols['규모']].min())
    max_mag = float(df[actual_cols['규모']].max())
    
    # 방어코드 A: NaN 값이 들어오는 경우 기본값 지정
    if np.isnan(min_mag) or np.isnan(max_mag):
        min_mag, max_mag = 0.0, 10.0
        
    # 방어코드 B: 데이터가 1개뿐이거나 모든 규모가 같아서 min과 max가 같을 때 슬라이더 고장 방지
    if min_mag == max_mag:
        min_mag = max(0.0, min_mag - 0.5)
        max_mag = max_mag + 0.5
    
    selected_mag = st.sidebar.slider(
        "지진 규모(Magnitudo) 선택",
        min_value=min_mag,
        max_value=max_mag,
        value=(min_mag, max_mag),
        step=0.1
    )
    
    # 슬라이더 필터 적용
    filtered_df = df[
        (df[actual_cols['규모']] >= selected_mag[0]) & 
        (df[actual_cols['규모']] <= selected_mag[1])
    ].copy()

    # 6. 핵심 지표 요약 (Metrics)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("현재 지도 표시 건수", f"{len(filtered_df)} 건")
    with col2:
        st.metric("최대 지진 규모", f"{filtered_df[actual_cols['규모']].max():.2f}" if len(filtered_df) > 0 else "0.00")
    with col3:
        st.metric("평균 진원 깊이", f"{filtered_df[actual_cols['진원깊이']].mean():.1f} km" if len(filtered_df) > 0 else "0.0 km")

    # 7. 지도 시각화 환경 설정 (Pydeck)
    st.subheader("🗺️ 실시간 지진 위험도 공간 매핑")
    st.markdown("💡 점의 크기가 **클수록**, 색상이 **붉을수록** 가상/예측 위험도가 높은 지역입니다.")
    
    mid_lat = filtered_df[actual_cols['위도']].dropna().mean()
    mid_lon = filtered_df[actual_cols['경도']].dropna().mean()

    def assign_color(row):
        risk = row['normalized_risk']
        r = int(risk * 255)
        g = int((1 - risk) * 255)
        b = 0
        return [r, g, b, 160]

    if len(filtered_df) > 0:
        filtered_df['color'] = filtered_df.apply(assign_color, axis=1)
        filtered_df['radius'] = (filtered_df['normalized_risk'] * 15000) + 3000

        layer = pdk.Layer(
            "ScatterplotLayer",
            filtered_df,
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_scale=1,
            radius_min_pixels=4,
            radius_max_pixels=40,
            get_position=f"[{actual_cols['경도']}, {actual_cols['위도']}]",
            get_radius="radius",
            get_fill_color="color",
            get_line_color=[255, 255, 255],
        )

        view_state = pdk.ViewState(
            latitude=mid_lat if not np.isnan(mid_lat) else 36.5,
            longitude=mid_lon if not np.isnan(mid_lon) else 127.5,
            zoom=2.5, 
            pitch=20,
        )

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "text": f"규모: {{{actual_cols['규모']}}}\n"
                        f"진원깊이: {{{actual_cols['진원깊이']}}}km\n"
                        f"위험도 점수: {{risk_score:.2f}}"
            }
        ))
    else:
        st.warning("선택한 조건에 맞는 지진 데이터가 존재하지 않습니다.")

    # 8. 하단 데이터 테이블 표출
    st.subheader("📊 샘플링된 지진 데이터 명세 (상위 100개 행)")
    display_cols = list(actual_cols.values()) + ['risk_score']
    st.dataframe(filtered_df[display_cols].head(100), use_container_width=True)

else:
    st.info("📢 웹 서비스를 시작하려면 프로젝트 폴더 안에 `earthquake.csv` 파일이 정상적으로 존재해야 합니다.")
