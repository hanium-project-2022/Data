import pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyproj as proj

def convert_coord(lat, lon, alt, transformer, return_type):
    # print(lat, lon, alt)
    tm_x, tm_y, tm_alt = transformer.transform(lat, lon, alt)
    # print(tm_x, tm_y, tm_alt)
    if return_type == 'tm_x':
        return tm_x
    elif return_type == 'tm_y':
        return tm_y
    elif return_type == 'tm_alt':
        return tm_alt

def main():
    ## CAN 및 GNSS 데이터 경로 설정
    base_path = pathlib.Path.cwd()
    can_path = list(base_path.glob("*/CAN/*"))
    can_path = sorted(can_path, key=lambda x: x.parent.parent.name)
    gnss_path = list(base_path.glob("*/GNSS/*"))
    gnss_path = sorted(gnss_path, key=lambda x: x.parent.parent.name)
    
    ## 저장 데이터 경로 설정
    result_paths = [path.parent / "Result" / path.name for path in base_path.glob("*/") if path.name.isnumeric()]
    for path in result_paths:
        path.mkdir(exist_ok=True, parents=True)
    result_paths = sorted(result_paths, key=lambda x: int(x.name))
    
    ## 모든 데이터 read하여 리스트로 저장
    print("Start reading files...")
    can_dfs = [pd.read_csv(path, low_memory=False) if not len(list(res_path.glob("*"))) else None for path, res_path in zip(can_path, result_paths)]
    gnss_dfs = [pd.read_csv(path, low_memory=False) if not len(list(res_path.glob("*"))) else None for path, res_path in zip(gnss_path, result_paths)]
    print("Reading files done.")
    # can_dfs = [pd.read_csv(can_path[0], low_memory=False)]
    # gnss_dfs = [pd.read_csv(gnss_path[0], low_memory=False)]    

    ## 필요한 Column들만 남기기
    can_dfs = [df[['timestamp2', 'LAT_ACCEL', 'LONG_ACCEL']].dropna() if df is not None else None for df in can_dfs]
    gnss_dfs = [df[['Timestamp', 'Latitude', 'Longitude', 'GPSMode', 'Altitude', 'Yaw', 'Pitch', 'Roll']] if df is not None else None for df in gnss_dfs]
    
    ## 좌표계 변환 객체 생성
    proj_5179 = 'epsg:5179' ## PCS (TM)
    proj_4326 = 'epsg:4326' ## GCS
    transformer = proj.Transformer.from_crs(proj_4326, proj_5179)    
    
    save_data = True
    for can_df, gnss_df, res_path in zip(can_dfs, gnss_dfs, result_paths):
        if len(list(res_path.glob("*"))):
            continue        
        print(f"Current: {res_path.name}")

        ## 서울 기준 GNSS 수집 시간을 UTC 기준 timestamp로 변환
        gnss_timestamp = pd.to_datetime(gnss_df['Timestamp'], format="%Y_%m_%d_%H_%M_%S_%f")
        gnss_timestamp = gnss_timestamp.dt.tz_localize('Asia/Seoul').dt.tz_convert('utc')
        gnss_timestamp = gnss_timestamp.values.astype(np.float64) / 10e8
        
        ## 겹치는 시간대만 남기기
        gnss_df['Timestamp'] = gnss_timestamp
        can_timestamp = can_df['timestamp2'].to_numpy()
        can_df = can_df[(can_df['timestamp2'] >= gnss_timestamp[0]) & (can_df['timestamp2'] <= gnss_timestamp[-1])].reset_index(drop=True)
        gnss_df = gnss_df[(gnss_df['Timestamp'] >= can_timestamp[0]) & (gnss_df['Timestamp'] <= can_timestamp[-1])].reset_index(drop=True)
        
        ## GNSS timestamp 기준으로 가장 가까운 timestamp의 CAN 데이터를 찾아 new_df에 concat
        gnss_timestamp = gnss_df['Timestamp'].to_numpy()
        new_df = []
        can_first = True
        can_columns = can_df.columns
        for timestamp in tqdm(gnss_timestamp):
            can_idx_df = (can_df['timestamp2']-timestamp).abs().sort_values()
            can_tmp_df = can_df.iloc[can_idx_df.index[0], :].to_numpy().reshape(1, -1)
            can_tmp_df = pd.DataFrame(can_tmp_df, columns=can_columns)        
            if can_first:
                new_df = can_tmp_df
                can_first = False
            else:
                new_df = pd.concat([new_df, can_tmp_df], ignore_index=True)
        
        ## CAN 데이터 timestamp 제거 후 CAN 데이터 및 GNSS 데이터 concat
        new_df = new_df.drop('timestamp2', axis=1)
        new_df = pd.concat([gnss_df, new_df], axis=1)

        ## GCS 좌표계 -> PCS 좌표계 변환 및 열 추가
        new_df['TM_X'] = new_df.apply(lambda x: convert_coord(x['Latitude'], x['Longitude'], x['Altitude'], transformer, return_type='tm_x'), axis=1)
        new_df['TM_Y'] = new_df.apply(lambda x: convert_coord(x['Latitude'], x['Longitude'], x['Altitude'], transformer, return_type='tm_y'), axis=1)
        new_df['TM_Altitude'] = new_df.apply(lambda x: convert_coord(x['Latitude'], x['Longitude'], x['Altitude'], transformer, return_type='tm_alt'), axis=1)

        if save_data:
            res_path = res_path / str(res_path.name + "_can_gnss_merged.csv")
            new_df.to_csv(res_path, index=False)


if __name__ == "__main__":
    main()