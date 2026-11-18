# MAICON 드론 영상 수집·상자·UGV 인식

공식 카메라 API에서 받은 JPEG를 저장하고, 상자와 UGV의 개수와 픽셀 위치를 OpenCV로 계산합니다. 연속된 프레임에서 가까운 UGV 관측에는 같은 임시 추적 번호를 부여합니다. 이번 버전의 출력 범위는 차체 검출과 위치 추적입니다. Python 3.10 이상이 필요합니다.

```sh
python -m pip install -r requirements.txt
```

`HOST`에는 주최측이 알려준 서버 주소를 입력합니다. 임무코드는 `MAICON_MISSION_CODE` 환경 변수나 터미널의 숨김 입력으로 제공합니다. 연습 모드는 `8100`의 `test1` 또는 `test2` 중 하나를, 본경기 모드는 `8000`의 `main` 영상 한 개를 사용합니다.

```sh
python capture_boxes.py --mode practice --origin http://HOST:8100 --camera test1 --roi 255 0 1010 720 --frames 6 --seconds 45
python capture_boxes.py --mode match --origin http://HOST:8000 --frames 8 --seconds 60
```

각 실행 결과는 `captures/` 아래에 저장됩니다. `.jpg`는 서버 원본, `.png`는 상자 윤곽과 번호를 표시한 이미지입니다. `frames.jsonl`에 캡처 시각, 원본 해시, 후보 수량과 픽셀 위치가 들어가고 `summary.json`에는 저장·분석 수와 종료 이유가 들어갑니다.

`frame-XXXX-ugvs.png`에는 UGV 윤곽과 임시 추적 번호가 표시됩니다. JSON에서 `observations`와 `observed_count`는 상자 결과이며, `ugvs`와 `ugv_count`는 UGV 결과입니다.

팀 코드에서는 `DroneFeed`로 JPEG를 받아 `analyze_jpeg`에 전달할 수 있습니다. `roi`는 원본 영상에서 경기장 영역의 `(left, top, right, bottom)` 픽셀입니다. 위 연습장 예시는 `test1`, `test2`의 1280×720 영상에서 검증한 영역입니다. `main`은 실제 경기장 영상에 맞춰 영역을 지정하세요. 결과는 원본 영상 기준 픽셀 좌표입니다.

```python
import os

from capture_boxes import analyze_jpeg
from drone_feed import DroneFeed

feed = DroneFeed(os.environ["MAICON_CAMERA_ORIGIN"], os.environ["MAICON_MISSION_CODE"], camera="test1")
frame = feed.fetch()
result = analyze_jpeg(frame.jpeg, roi=(255, 0, 1010, 720))
print(result["observed_count"], result["observations"])
print(result["ugv_count"], result["ugvs"])
```

대비를 보정한 영상에서 사각형 윤곽을 찾고 원본의 테두리 인쇄로 확인합니다. 붙어 있는 상자는 각각의 인쇄 경계를 확인해 따로 셉니다. 표시 이미지에서 검출 위치를 확인할 수 있습니다.

## UGV 추적과 실사진 대조

검출기는 차체의 윤곽과 수광 돔으로 보이는 밝은 특징을 이용합니다. 벽이나 상자에 차체 윤곽이 붙어 보이면 두 수광부 후보 주변의 차체를 확인합니다. 같은 수광부를 다시 잡은 윤곽은 중복 집계에서 제외합니다. 각 관측에 `bbox_px`, `center_px`, 검출 점수가 기록됩니다. 좌표는 ROI를 사용해도 원본 영상 기준 픽셀입니다.

`visual-0001` 형식의 `track_id`는 해당 촬영 세션에서 연속 관측을 연결하는 임시 번호입니다. 교차가 불분명하거나 기체가 가려지거나 촬영 간격이 길어지면 번호가 새로 부여될 수 있습니다. 단일 사진에는 해당 사진의 표시 번호만 붙습니다.

저장된 항공 사진은 서버 연결 없이 다음 명령으로 검사합니다. 출력 디렉터리는 새 이름을 사용합니다.

```sh
python analyze_aerial_image.py aerial-photo.jpg --output review/photo-01 --roi 255 0 1010 720
```

`ugvs.png`와 `boxes.png`에서 위치를 대조하고, `analysis.json`에서 개수와 좌표를 확인할 수 있습니다. UGV만 확인할 때는 `--ugv-only`를 붙입니다.

## 검사와 공유 범위

`python -m unittest discover -v`로 검출·추적·수집 검사를 실행합니다. 기존 연습장 사진 13장은 검출 26→30건, 누락 5→1건, 오탐 1→0건이었습니다. 변경에 쓰지 않은 연습장 사진 10장은 검출 13→14건, 누락 3→2건, 오탐 0건을 유지했습니다. 두 묶음은 같은 연습장의 촬영본입니다. 평가 기준은 `validation_summary.json`에 있습니다.

원본 촬영 사진·영상·키·임무 코드는 Git에 포함하지 않습니다. 공유되는 것은 코드와 합성 검사, 문서 및 비식별 검증 요약입니다.
