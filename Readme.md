현재 통합상황
- both_jsm
- feature/both -> 카메라 람다분할될거 적용
- feature/head -> 단일 암 선택기능추가, 헤드모터적용유무선택기능추가, 이거하면서 테스크 모션 수정도 적용함
- feature/nohead_jsm -> 헤드를 사용 안하겠다고 할 때 자동으로 카메라 브라켓 기반의 설계값을 사용 및 헤드관련 체크박스 지우도록 적용
- feature/qp-solver -> qp solver 적용, 최적화관련 기능들 분리하고 ui에 가우스뉴턴으로할지, qp로 할지 선택하는 기능 추가

지워도되는 브렌치
- dev : 진짜 오래된거. 어차피 꼬여서 버려야되는 브렌치
- feature/nohead : 현재 feature/nohead_jsm에 통합되어있음. 삭제해도 됨
- feature/prototype : 별다른 기능차이점 없음. 삭제해도 됨
- feature/7_parameter : 옛날브렌치. 바로삭제가능


# Camera Calibration


## Setup
- 보정에 있어 필요한 라이브러리가 많으므로, 가상공간을 만들어서 설치 후 사용하는것을 권장

```bash
python3 -m venv .venv 
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

- 가상환경 종료시에는 `deactivate` 입력

## object

## Camera & Bracket Setting

## Calibration Method

## Caution