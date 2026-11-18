# MAICON 드론 영상 수집과 상자 인식

공식 카메라 API에서 받은 JPEG를 저장하고, 인쇄된 상자 후보의 수량과 픽셀 위치를 OpenCV로 계산합니다. Python 3.10 이상이 필요합니다.

```sh
python -m pip install -r requirements.txt
```

`HOST`에는 주최측이 알려준 서버 주소를 입력합니다. 임무코드는 `MAICON_MISSION_CODE` 환경 변수나 터미널의 숨김 입력으로 제공합니다. 연습 모드는 `8100`의 `test1` 또는 `test2` 중 하나를, 본경기 모드는 `8000`의 `main` 영상 한 개를 사용합니다.

```sh
python capture_boxes.py --mode practice --origin http://HOST:8100 --camera test1 --roi 255 0 1010 720 --frames 6 --seconds 45
python capture_boxes.py --mode match --origin http://HOST:8000 --frames 8 --seconds 60
```

각 실행 결과는 `captures/` 아래에 저장됩니다. `.jpg`는 서버 원본, `.png`는 상자 윤곽과 번호를 표시한 이미지입니다. `frames.jsonl`에 캡처 시각, 원본 해시, 후보 수량과 픽셀 위치가 들어가고 `summary.json`에는 저장·분석 수와 종료 이유가 들어갑니다.

팀 코드에서는 `DroneFeed`로 JPEG를 받아 `analyze_jpeg`에 전달할 수 있습니다. `roi`는 원본 영상에서 경기장 영역의 `(left, top, right, bottom)` 픽셀입니다. 위 연습장 예시는 `test1`, `test2`의 1280×720 영상에서 검증한 영역입니다. `main`은 실제 경기장 영상에 맞춰 영역을 지정하세요. 결과는 원본 영상 기준 픽셀 좌표입니다.

```python
import os

from capture_boxes import analyze_jpeg
from drone_feed import DroneFeed

feed = DroneFeed(os.environ["MAICON_CAMERA_ORIGIN"], os.environ["MAICON_MISSION_CODE"], camera="test1")
frame = feed.fetch()
result = analyze_jpeg(frame.jpeg, roi=(255, 0, 1010, 720))
print(result["observed_count"], result["observations"])
```

대비를 보정한 영상에서 사각형 윤곽을 찾고 원본의 테두리 인쇄로 확인합니다. 붙어 있는 상자는 각각의 인쇄 경계를 확인해 따로 셉니다. 표시 이미지에서 검출 위치를 확인할 수 있습니다.
