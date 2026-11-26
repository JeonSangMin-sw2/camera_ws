# 헤드 카메라와 양팔 플랜지 마커를 이용한 모바일 휴머노이드 관절 영점의 단계적 보정

**Stepwise Calibration of Joint Zero Offsets for a Wheeled Humanoid Using a Head Camera and Markers on Both Arms**

> 리뷰용 원문입니다. 실제 원고는 Word(docx)이며 LaTeX 원본은 없습니다. 수식은 LaTeX 표기로 옮겼고, 본문의 $x$, $_{k}$, $^{k}$ 표기는 원고에서 이탤릭·아래첨자·위첨자이고 [n]은 위첨자 인용입니다.

**Abstract**: Small joint zero offsets shift the end effector by millimeters, so programs and learned policies do not transfer between units of the same robot. We calibrate the 16 joint zero offsets of the wheeled humanoid RB-Y1 (7 joints per arm and 2 head joints) using only its head camera and AprilTag markers on both tool flanges. Because joints with nearly parallel axes cannot be separated in one whole-body fit, the joints are calibrated in three steps. Step 1 estimates the elbow and wrist zeros from the geometry of encoder-phased circles fitted to single-joint sweeps. Step 2 sweeps only the head while both arms hold their markers still. Step 3 solves the remaining joints with a bounded quadratic program that anchors only unidentifiable directions and damps only the common mode of the two shoulder joints. Over six independent resets, the median marker error at the ready pose fell from 15.42 mm to 0.11 mm.

**Keywords**: Joint offset calibration, Wheeled humanoid, Circle point analysis, Fiducial marker, Parameter coupling

## 1. 서  론

본 연구는 모바일 휴머노이드(wheeled humanoid) RB-Y1의 양팔 7축(총 14축)과 헤드 2축(pan, tilt)을 합한 16축 관절 영점을 헤드 카메라와 마커만으로 보정하는 방법을 다룬다. 생산 과정의 미세한 조립 공차나, 학습 데이터 취득·실제 운용 중의 충돌 사고로 관절 출력단 체결부가 조금만 틀어져도 엔코더가 읽는 각과 실제 관절각 사이에 작은 영점 차이가 생긴다. 이 각도 오차는 관절에서 끝단까지의 거리에 비례해 위치 오차로 커져, 팔을 편 자세(첫 관절 축에서 끝단까지 약 620 mm)에서 첫 관절이 0.2°만 틀어져도 끝단은 약 2.2 mm, 모든 팔 관절이 0.2°씩 틀어지면 약 4.6 mm 벗어난다. 그래서 같은 기종이라도 규칙 기반으로 짠 같은 작업이 로봇마다 다른 곳에 닿고, VLA(vision-language-action) 모델처럼 한 로봇의 시연 데이터로 학습한 정책은 다른 로봇에서 추론 결과가 달라질 수 있다.

관절 영점을 보정하는 대표적인 방법은 레이저 트래커 같은 외부 측정기로 끝단 위치를 재는 기구학 보정[1]이며, 정확하지만 장비가 비싸고 측정 공간이 필요하다. 카메라로 보정하는 방법으로는 카메라와 로봇 사이 변환을 구하는 핸드-아이 보정[2]과, 관절 오프셋과 센서 위치를 함께 추정하는 자가 보정이 PR2[3], NAO[4], Agile Justin[5]에서 연구되었다. 그러나 보정할 관절이 많으면 여러 파라미터가 마커 자세에 거의 같은 변화를 주어 서로 구분되지 않는데(파라미터 커플링), 이를 하나의 최적화로 동시에 풀면 개별 값이 정해지지 않는다. 정리하면 기존 방법은 비싼 외부 장비가 필요하거나, 파라미터 커플링을 따로 다루지 않는다.

헤드 카메라 하나로 16축을 한꺼번에 풀면 손목 끝 관절과 헤드 tilt처럼 축이 나란한 관절은 Jacobian 열이 거의 평행해 따로 식별되지 않는다. 이는 기구학 보정의 식별성 문제에 해당하며[1], 보통은 커플링된 쌍의 한쪽을 별도 측정이나 사전값으로 정해 다룬다. 본 연구도 이 관점에서 툴 플랜지 마커와 헤드 카메라만으로 16축의 보정 순서를 제시하며, 양팔을 함께 쓰는 점을 보정에 활용한다. Step 1은 단일 관절 회전의 원 기하로 엘보우·손목 영점을 잰다. Step 2는 양팔 마커를 고정한 채 헤드만 움직여 헤드 영점과 카메라 자세를 맞춘다. Step 3은 남은 관절을 식별 불가 방향에만 앵커를 건 OSQP[6] 기반 전신 QP로 풀며, 양팔 첫 관절의 공통 성분만 헤드 tilt와 구분되지 않는다는 좌우 대칭을 이용한다.

## 2. RB-Y1과 보정 시스템

### 2.1 RB-Y1 관절 구성

RB-Y1(v1.2)은 이동 베이스 위에 토르소, 7축 양팔, 2축 헤드를 얹은 모바일 휴머노이드다. 팔 관절은 어깨부터 J0(pitch), J1(roll), J2(yaw), J3(pitch, 엘보우), J4(yaw), J5(pitch), J6(yaw) 순이고, J0 축은 pitch 방향에서 20° 기울어 있다. 헤드는 pan(yaw)과 tilt(pitch)로 움직인다. J2–J3–J4와 J4–J5–J6은 yaw–pitch–yaw 구조라, 가운데 pitch 관절(J3, J5)이 0°이면 양옆 yaw 축이 한 방향에 놓이고, J6은 J5와 직교한다. 이 구조가 3.1절의 기하 조건이 된다. 반면 헤드 카메라로 볼 때 Step 3 준비 자세에서 J6 축은 헤드 tilt 축과 나란하고(방향 코사인 |cos| 1.000), J0 축은 20° 기울어 고정돼 있지만 0.940으로 tilt 방향 성분이 대부분이며, J1과 헤드 pan도 0.909로 나란해, 이런 쌍은 오프셋이 마커 자세에 거의 같은 변화를 주어 커플링된다.

### 2.2 보정 시스템 구성

[Fig. 1]과 같이 양팔 툴 플랜지에 80 mm AprilTag[7](36h11, 오른쪽 8번, 왼쪽 7번)를 붙인 정밀 브라켓을, 헤드에 RealSense D405를 체결한다. 카메라 내부 파라미터는 ChArUco 보드를 16장 촬영해 보정한다. 마커 자세는 AprilTag 검출과 PnP(IPPE)로 구하고, 검출 파라미터 조정과 왜곡을 포함한 내부 파라미터 보정으로 잡음을 줄인다.

*(그림: fig1_zero.png)*

**[Fig. 1]** Mounting on RB-Y1 (zero pose, simulation)

## 3. 단계적 보정

[Fig. 2]는 단계 흐름과 각 관절의 영점을 정하는 단계이고, 2.1절의 커플링 쌍은 [Table 1]처럼 한쪽을 앞 단계 값이나 설계값에 고정하거나 앵커를 건다. 카메라 x축 회전과 tilt, 브라켓 yaw와 J6은 구분되지 않으므로 카메라 회전과 브라켓 yaw는 정한 값에 둔다. 앞 단계에서 대략 잡은 값(헤드 영점, J6)은 뒤 단계의 초기값이자 기준이 되고, 뒤 단계가 그 값에서 출발해 남은 오차를 더 보정한다.

*(그림: fig3.png)*

**[Fig. 2]** Calibration steps and the joints each step sets

**[Table 1]** COUPLED PARAMETERS IN STEP 3

| Parameter | From | In Step 3 | Coupled |
|---|---|---|---|
| J3, J5 | Step 1 | Fixed, 10{sup:7} | — |
| J6 | Step 1 | Solved | Tilt |
| Head tilt | Step 2 | Anchor 2×10{sup:5}, ±2° | J6, J0 |
| Head pan | Step 2 | Anchor 5, ±15° | J1 |
| J0 (R + L) | — | Damping 2000 | Tilt |
| Camera rx, ry | Step 2 | Fixed | Tilt, pan |
| Camera pos. | Step 2 | Design range | — |
| Bracket pose | Step 1 | Fixed | J6 (yaw) |

### 3.1 Step 1: 원 기하 기반 엘보우·손목 영점

Step 1은 관절 하나를 돌리는 스윕 동안 프레임마다 마커 위치 $p_{k}$와 관절각 $q_{k}$를 기록하고, 식 (1)의 원 모델을 Huber 손실로 피팅한다.

$$ p_k = c + r\,[\cos\theta_k\,u + \sin\theta_k\,(n\times u)] \qquad (1) $$

$c$, $r$은 원의 중심과 반지름, $n$은 회전축(원의 법선), $u$는 원 평면의 기준 방향, $s$는 회전 방향 부호이며 위상은 $θ_{k}$ = $s$($q_{k}$ − $q_{0}$)이다. 위상을 관절각에 묶으면 자유 원 피팅이 불안정해지는 짧은 호[8]에서도 반지름이 안정된다. 마커 측정에 남은 잡음이 없다고 가정하면, 관절 하나를 돌렸을 때 마커는 그 축 둘레로 원을 그리고 영점이 맞으면 여러 원의 기하(중심, 법선, 반지름)는 설계와 같아야 한다. 이 가설에서 다음 세 조건으로 영점을 잰다. 외부 측정기로 축을 재는 circle point analysis[1]와 달리 헤드 카메라로 축 사이의 관계를 재므로, 카메라 외부 파라미터 오차는 원 사이의 차이에서 대부분 상쇄된다.

조건 ① 원 일치(J3, J5): 대상 관절을 사이에 둔 yaw 축 A, B가 같은 방향이 되는 자세에서 A와 B를 각각 돌려 두 원을 얻는다(J3은 A = J2, B = J4, J5는 A = J4, B = J6). 대상 관절의 영점이 맞으면 두 원은 겹치고, 틀리면 대상 축 둘레로 두 중심이 벌어진다. 보정량 δ$q$는 식 (2)로 구한다(식 (5)의 영점 오프셋과 δ$q$ = −Δ$q$ 관계).

$$ \delta q = -\arcsin\!\left(\frac{(c_B^{\perp}-c_A^{\perp})\cdot(a\times\hat v_A)}{\lVert v_B\rVert}\right) \qquad (2) $$

$c_{A}^{⊥}$, $c_{B}^{⊥}$는 대상 축 $a$에 수직인 평면으로 사영한 두 원의 중심, $v_{A}$, $v_{B}$는 대상 관절 원점에서 각 중심까지의 벡터, $v̂_{A}$는 $v_{A}$의 단위 벡터다. [Fig. 3](a)는 J3을 잴 때 J2와 J4를 각각 ±20° 돌리는 동작이고, (b)는 그때 기록한 마커 궤적(점)과 피팅한 원(선)이다.

조건 ② 직교 축의 방향(J6): J6과 J5 스윕 원의 법선을 마커 좌표로 옮기면, J6 영점이 맞을 때 J6 축 $n_{6}$ 둘레에서 본 J5 축의 방향은 설계 J5 축 $e_{5}^{nom}$과 같아야 하며, 보정량은 식 (3)이다.

$$ \delta q_6 = \operatorname{atan2}\!\big(\hat n_5^{\perp}\cdot(n_6\times e_5^{\mathrm{nom}}),\ \hat n_5^{\perp}\cdot e_5^{\mathrm{nom}}\big) + q_6^{\mathrm{enc}} \qquad (3) $$

$n̂_{5}^{⊥}$는 측정한 J5 축을 $n_{6}$에 수직인 평면에 사영한 단위 벡터, $q_{6}^{enc}$는 스윕 시작 J6 엔코더 각이다. J6 영점은 브라켓의 J6 동축 회전(yaw)과 구분되지 않아 Step 3이 이 값에서 다시 푼다.

조건 ③ 원 반지름(브라켓 위치): 원 반지름은 축에서 마커까지의 거리와 같다. 마커 위치를 엔드이펙터 좌표 ($x$, $y$, $z$)로 두면 J6 축은 $z$, J5 축은 $y$ 방향이고 J5 축은 J6 축 방향으로 $L$만큼 떨어져 있어, 엔코더 위상으로 피팅한 반지름 $r_{6}$, $r_{5}$는 식 (4)를 만족한다.

$$ r_6^2 = x^2 + y^2,\quad r_5^2 = x^2 + (z+L)^2 \qquad (4) $$

$y$는 설계값을 신뢰해 고정하고 $x$, $z$를 조립 공차 안에서 푼다. 스윕, 피팅, 보정량 계산, 적용은 [Table 2]의 기준을 모두 만족할 때까지 반복하고, 목표 근처 추정값의 평균을 보정값으로 쓴다. $s$는 추정값의 표본 표준편차이고, 잡음 하한 $s_{min}$은 우연히 가까운 두 값으로 통과하는 것을 막는다.

**[Table 2]** PASS CRITERIA OF STEP 1

| Item | Criterion |
|---|---|
| Iterations, step | ≤ 8, ≤ 5.0° per iteration |
| Estimates used | Within 0.3° of target, {i:N} ≥ 2 |
| Standard error | max({i:s}, {i:s}{sub:min}) / √{i:N} ≤ 0.06° |
| {i:s}{sub:min} (J5, J6, J3) | 0.03°, 0.06°, 0.12° |

### 3.2 Step 2: 헤드-카메라

같은 마커 자세를 로봇 경로와 비전 경로로 계산해 자세 $k$의 잔차를 식 (5)로 정의한다.

$$ e_k(x) = \log\!\Big(\big(T_{t5,c}(x)\,T_{c,m}^{(k)}\big)^{-1}\,T_{t5,m}(q_k+\Delta q)\Big) \qquad (5) $$

t5는 두 경로의 공통 기준인 토르소 최상단 링크, $T_{t5,m}$($q_{k}$ + Δ$q$)는 관절각과 영점 오프셋으로 계산한 마커 자세, $T_{t5,c}$($x$)는 헤드 순기구학과 카메라 외부 파라미터로 계산한 카메라 자세, $T_{c,m}^{(k)}$는 카메라가 잰 마커 자세, $x$는 추정 변수(관절 오프셋, 헤드 영점, 카메라 자세)이며, log는 두 자세의 차이를 위치 3, 방향 3의 6차원 벡터로 바꾼다. Step 2는 양팔을 고정해 두 마커를 표지로 삼고 헤드만 tilt ±10°, pan ±15°로 왕복하며, 식 (5)의 위치 성분으로 헤드 영점, 카메라 위치, 광축 회전을 범위 제약 최소제곱으로 풀고, 카메라-엔코더 지연도 함께 맞춘다.

### 3.3 Step 3: 전신 QP

식 (5)를 64자세에 대해 가중 최소제곱으로 풀고, 선형화한 정규방정식을 식 (6)의 목적함수와 식 (7)의 제한조건을 갖는 QP로 만들어 OSQP[6]로 반복한다.

$$ \min_{\Delta x}\ \lVert J\Delta x + e\rVert_W^2 + \sum_i w_i\,(x_i + \Delta x_i - x_i^{\mathrm{ref}})^2 \qquad (6) $$

$$ x^{\mathrm{lo}} \le x + \Delta x \le x^{\mathrm{hi}} \qquad (7) $$

$x$는 양팔 관절 오프셋, 헤드 영점, 카메라 위치를 모은 추정 변수, Δ$x$는 한 번의 갱신량, $e$는 모든 자세의 잔차 $e_{k}$를 쌓은 벡터, $J$는 $e$를 $x$로 미분한 Jacobian, $W$는 자세 잔차 6성분의 측정 잡음 역분산(방향 표준편차 하한 0.3°)이다. 둘째 항은 [Table 1]의 고정·앵커 항으로 $x_{i}^{ref}$는 앞 단계 값, $w_{i}$는 가중이며, 식 (7)의 $x^{lo}$, $x^{hi}$는 [Table 1]의 앵커 범위와 카메라 위치의 설계 범위다. 헤드 tilt 오차는 양팔 J0이 같은 방향으로 돌아 상쇄할 수 있어, 양팔 J0의 합에만 같은 꼴의 댐핑 항을 더하고 좌우 반대 성분은 데이터로 푼다.

## 4. 실험 결과

[Fig. 3], [Fig. 4]와 [Table 3]이 RB-Y1 v1.2 실기의 보정 전후 비교다. 준비 자세 마커 오차는 같은 측정을 관절 오프셋 적용 전후의 모델 예측과 비교한 값이며(카메라·브라켓은 둘 다 보정값), 영점 리셋 6회마다 첫 보정 실행의 양팔 값(n = 12)의 중앙값이다.

**[Table 3]** BEFORE AND AFTER CALIBRATION

| Metric | Before | After |
|---|---|---|
| J3 center distance (right) | 7.18 mm | 0.08 mm |
| J5 center distance (right) | 1.99 mm | 0.34 mm |
| Ready pose, median (n = 12) | 15.42 mm / 2.45° | 0.11 mm / 0.19° |

**[Table 4]** ESTIMATED JOINT ZERO OFFSETS Δq (DEG)

| Arm | J0 | J1 | J2 | J3 | J4 | J5 | J6 |
|---|---|---|---|---|---|---|---|
| Right | 0.04 | 1.44 | 0.25 | 1.84 | 0.00 | 0.67 | −0.07 |
| Left | −0.05 | −2.44 | 0.37 | 2.26 | −0.48 | 0.80 | 0.73 |

**[Table 5]** STEPWISE VS. ONE-SHOT OSQP

| Metric | Stepwise | One-shot |
|---|---|---|
| Residual (median) | 0.10-0.35 mm | 0.09-0.24 mm |
| Head tilt offset | −1.6 to 0.2° | −11.4 to 10.2° |
| Hand shift at check pose | 0 (ref.) | 1.2-5.7 mm |

*(그림: fig3_rot.png)*

**[Fig. 3]** Right elbow J3: (a) J2, J4 swept ±20° (simulation), (b) fitted circles of the J2 (red) and J4 (blue) sweeps

*(그림: fig4.png)*

**[Fig. 4]** Check pose before and after: (a) front, (b) side view

[Fig. 4]에서는 카메라를 거치지 않고도 두 마커 판이 보정 후 한 면으로 맞는다. [Table 4]는 한 실행의 관절별 보정량으로, 양팔 J1과 J3이 가장 크다. [Table 5]는 같은 64자세로 모든 변수를 앵커 없이 한 번에 푼 OSQP와의 비교다(리셋 6회 중 수렴한 5회). 잔차(실행별 중앙값)는 비슷하지만 헤드 tilt가 카메라 x축 회전과 구분되지 않아 그 범위 끝(±10°)까지 밀리며, 실물로 맞은 단계식 결과를 기준으로 계산하면 확인 자세에서 손이 1.2-5.7 mm 어긋나는 것이 확인되었다.

## 5. 결  론

RB-Y1의 16축 관절 영점을 세 단계로 나누어 커플링되는 관절 쌍을 단일 관절 측정과 양팔 좌우 대칭으로 분리했다. 한 번에 푼 OSQP는 잔차가 비슷해도 손이 수 mm 어긋났다. 한계로, 관절과 구분되지 않는 카메라 x축(roll) 회전과 마커 브라켓의 회전 오차는 보정하지 못했다. 향후 이 형식을 토르소 관절까지 넓히고 외부 측정기로 검증할 예정이다.

## References

[1] J. M. Hollerbach and C. W. Wampler, “The calibration index and taxonomy for robot kinematic calibration methods,” *International Journal of Robotics Research*, vol. 15, no. 6, pp. 573-591, 1996.
[2] R. Y. Tsai and R. K. Lenz, “A new technique for fully autonomous and efficient 3D robotics hand/eye calibration,” *IEEE Transactions on Robotics and Automation*, vol. 5, no. 3, pp. 345-358, 1989.
[3] V. Pradeep, K. Konolige, and E. Berger, “Calibrating a multi-arm multi-sensor robot: A bundle adjustment approach,” *Experimental Robotics*, Springer Tracts in Advanced Robotics, vol. 79, Springer, 2014, pp. 211-225.
[4] D. Maier, S. Wrobel, and M. Bennewitz, “Whole-body self-calibration via graph-optimization and automatic configuration selection,” *IEEE International Conference on Robotics and Automation*, Seattle, USA, pp. 5662-5668, 2015.
[5] J. Tenhumberg, D. Winkelbauer, D. Burschka, and B. Bäuml, “Self-contained calibration of an elastic humanoid upper body with a single head-mounted RGB camera,” *IEEE-RAS International Conference on Humanoid Robots*, Ginowan, Japan, 2022.
[6] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, “OSQP: An operator splitting solver for quadratic programs,” *Mathematical Programming Computation*, vol. 12, no. 4, pp. 637-672, 2020.
[7] J. Wang and E. Olson, “AprilTag 2: Efficient and robust fiducial detection,” *IEEE/RSJ International Conference on Intelligent Robots and Systems*, Daejeon, Korea, pp. 4193-4198, 2016.
[8] N. Chernov and C. Lesort, “Least squares fitting of circles,” *Journal of Mathematical Imaging and Vision*, vol. 23, pp. 239-252, 2005.
