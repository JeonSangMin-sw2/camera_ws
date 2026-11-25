# 헤드 카메라와 손목 플랜지 마커를 이용한 양팔 로봇 관절 영점의 단계적 보정

**Stepwise Calibration of Joint Zero Offsets for a Dual-Arm Robot Using a Head Camera and Flange-Mounted Markers**

> 리뷰용 원문입니다. 실제 원고는 Word(docx)이며 LaTeX 원본은 없습니다. 수식은 LaTeX 표기로 옮겼고, 본문의 $x$, $_{k}$, $^{k}$ 표기는 원고에서 이탤릭·아래첨자·위첨자이고 [n]은 위첨자 인용입니다.

**Abstract**: Joint zero offsets of a robot drift with long-term use and collisions and differ between units, so a policy learned on one robot can degrade on another unit of the same model. We calibrate the 16 joint offsets of the dual-arm robot RB-Y1 (7 joints per arm and 2 head joints) using only the head camera and AprilTag markers on flange brackets. The key idea is to partition the joints by identifiability. Step 1 rotates single elbow and wrist joints, fits encoder-phased circles to the marker trajectory, and derives zero offsets from three geometric conditions: coincident circles of parallel axes, the direction of an orthogonal axis, and radii equal to design distances. Step 1.5 solves the head offsets, camera pose and latency, and Step 2 solves the remaining joints with a bounded quadratic program that anchors only unidentifiable directions. Over six independent resets, the median marker error at the ready pose fell from 15.42 mm to 0.11 mm.

**Keywords**: Joint offset calibration, Redundant manipulator, Circle point analysis, Fiducial marker, Dual-arm robot

## 1. 서  론

본 연구는 바퀴형 양팔 로봇 RB-Y1(v1.2)의 양팔 7축(총 14축)과 헤드 2축(pan, tilt)을 합한 16축 관절 영점을 헤드 카메라와 마커만으로 보정하는 방법을 다룬다. 로봇을 오래 쓰거나 충돌을 겪으면 관절 원점이 조금씩 밀리고, 생산 과정에서도 로봇마다 작은 영점 차이가 생긴다. 관절의 각도 오차는 관절에서 끝단까지의 거리에 비례해 끝단 위치 오차로 커지므로, RB-Y1의 확인 자세에서 팔의 각 관절이 0.2°씩만 틀어져도 끝단 위치가 약 2~3 mm 벗어난다(J0 하나만으로 1.2 mm). 그래서 같은 기종이라도 규칙 기반으로 짠 같은 작업이 로봇마다 다른 곳에 닿고, VLA(vision-language-action) 모델처럼 한 로봇의 시연 데이터로 학습한 정책은 다른 로봇에서 추론 결과가 달라질 수 있다. 로봇을 바꿔도 같은 프로그램과 모델을 쓰려면 로봇마다 관절 오프셋을 현장에서 손쉽게 보정할 수 있어야 한다.

관절 영점을 보정하는 대표적인 방법은 레이저 트래커 같은 외부 측정기로 끝단 위치를 재어 기구학 파라미터를 추정하는 기구학 보정[3]이다. 정확도는 높지만 장비가 비싸고 측정 공간이 필요해, 현장의 여러 로봇에 적용하기 어렵다. 카메라를 쓰는 핸드-아이 보정[1, 2]은 저렴하지만 로봇 기구학이 정확하다고 가정하고 카메라와 로봇 사이 변환만 구하므로 관절 영점은 다루지 않는다. 로봇의 카메라로 손에 단 보정판이나 마커를 보며 관절 오프셋과 센서 위치를 함께 추정하는 자가 보정도 PR2[4], NAO[5], Agile Justin[6]에서 연구되었으나, 이들은 모든 파라미터를 하나의 최적화로 동시에 풀므로 서로 구분되지 않는 파라미터가 섞이면 개별 값이 정해지지 않는다. 본 방법은 툴 플랜지 마커와 헤드 카메라만으로 저렴하게 구성하면서, 동시 최적화로 구분되지 않는 관절을 단일 관절 회전 측정으로 먼저 확정한 뒤 나머지를 푼다. 따라서 커플링된 관절 쌍의 한쪽을 별도 측정으로 정해 분리할 수 있다는 장점이 있다.

본 논문의 핵심은 16축을 어떻게 나누어 어떤 순서로 보정하는가이다. 헤드 카메라 하나로 16축을 한꺼번에 풀면 J6과 헤드 tilt처럼 축이 나란한 관절들은 오프셋이 잔차에 주는 영향(야코비안 열)이 거의 평행해 따로 식별되지 않는다. 이를 파라미터 커플링이라 하며, 기구학 보정의 식별성 문제에 해당한다[3]. 그래서 커플링되는 쌍은 한쪽을 별도 측정으로 먼저 정해 고정하거나 앵커를 건다. 즉 (1) 단일 관절 회전의 원 기하로 엘보우·손목 영점을 먼저 확정하고(Step 1), (2) 헤드와 카메라를 따로 맞춘 뒤(Step 1.5), (3) 나머지 관절을 식별 불가 방향에만 앵커를 건 OSQP[7] 기반 전신 QP로 푼다(Step 2). 사전 보정 방법은 3장에서 설명한다. 특히 Step 1은 외부 기준 없이 축 사이의 기하만으로 영점을 구하므로, 7자유도 여자유도 매니퓰레이터의 오프셋 보정 방법으로 볼 수 있다.

## 2. 보정 구성과 단계

### 2.1 장착 구성

실험은 RB-Y1 v1.2 실기에서 수행했다. [Fig. 1]과 같이 양팔 툴 플랜지에 80 mm AprilTag[8](36h11, 오른쪽 8번, 왼쪽 7번)를 붙인 정밀 브라켓을 체결하고, 헤드에 RealSense D405를 카메라 브라켓으로 체결한다. 카메라 내부 파라미터는 A4 ChArUco 보드를 16장 촬영해 보정한다[9].

*(그림: fig1_new.png)*

**[Fig. 1]** Calibration kit and its mounting on RB-Y1: AprilTag brackets on the right (ID 8) and left (ID 7) tool flanges and a RealSense D405 on the head

### 2.2 단계별 관절 분할

[Fig. 2]는 단계 흐름과 각 관절의 영점을 정하는 단계다. 단계를 나눈 기준은 식별성이다. Step 2 준비 자세에서 헤드 카메라 기준으로 축 정렬도 |cos|를 구해 커플링되는 쌍을 찾고(J6·J0와 tilt 1.000, 0.940, J1과 pan 0.909), 쌍마다 한쪽을 앞 단계 값이나 설계값에 고정하거나 앵커를 걸었다([Table 1]). 카메라 x축 회전과 tilt, 브라켓 yaw와 J6은 완전히 커플링되어 구분되지 않으므로, 이를 감안해 카메라 회전과 브라켓 yaw는 정한 값에 두고 보정한다.

**[Table 1]** COUPLED PARAMETERS AND THEIR HANDLING IN STEP 2

| Parameter | From | In Step 2 | Coupled |
|---|---|---|---|
| J3, J5 | Step 1 | Fixed, 1e7 | — |
| J6 | Step 1 | Solved | Tilt |
| Head tilt | Step 1.5 | Anchor 2e5, ±2° | J6, J0 |
| Head pan | Step 1.5 | Anchor 5, ±15° | J1 |
| J0 (R + L) | — | Damping 2000 | Tilt |
| Camera rx, ry | Step 1.5 | Fixed | Tilt, pan |
| Camera pos. | Step 1.5 | Design range | — |
| Bracket pose | Step 1 | Fixed | J6 (yaw) |

*(그림: fig3.png)*

**[Fig. 2]** Joint partition: each joint is colored by the step that sets its zero offset

## 3. Step 1: 원 기하 기반 관절 영점 보정

### 3.1 마커 측정과 가설

마커 데이터는 헤드 카메라 영상에서 AprilTag를 검출하고 PnP(IPPE)로 마커 자세를 구해 얻으며, 두 해 중 재투영 오차가 작은 해를 쓴다. 측정 잡음은 검출 함수의 파라미터 조정(서브픽셀 코너 보정 등)과 렌즈 왜곡을 포함한 내부 파라미터 보정으로 줄이고, 헤드를 움직이는 Step 1.5에서는 카메라-엔코더 지연도 맞춘다. Step 1은 관절 하나를 돌리는 스윕 동안 프레임마다 마커 위치 $p_{k}$와 관절각 $q_{k}$를 함께 기록하고, 식 (1)의 원 모델을 Huber 손실로 피팅한다.

$$ p_k = c + r\,[\cos\theta_k\,u + \sin\theta_k\,(n\times u)] \qquad (1) $$

$c$, $r$은 원의 중심과 반지름, $n$은 회전축(원의 법선), $u$는 원 평면의 기준 방향, $s$는 회전 방향 부호이며 위상은 $θ_{k}$ = $s$($q_{k}$ − $q_{0}$)이다. 위상을 관절각에 묶으면 짧은 호에서도 반지름이 안정된다. 법선은 설계 축과 같은 쪽($n$·$n_{nom}$ > 0)으로 정한다. 마커 측정에 남은 잡음이 없다고 가정하면, 관절 하나를 돌렸을 때 마커는 그 축 둘레로 원을 그리고 영점이 맞으면 여러 원의 기하(중심, 법선, 반지름)는 설계와 같아야 한다. 이 가설에서 다음 세 조건으로 영점을 잰다. 단일 관절 회전으로 축을 재는 것은 circle point analysis[3]와 같지만, 본 방법은 외부 측정기 대신 헤드 카메라로 축 사이의 관계를 재며, 비교하는 원들을 같은 카메라로 재므로 카메라 외부 파라미터 오차는 원 사이의 차이에서 대부분 상쇄된다.

### 3.2 조건 ①: 같은 방향 두 축의 원 일치

대상 관절을 사이에 둔 두 축 A, B가 설계상 같은 방향이 되는 자세에서 A와 B를 각각 돌려 두 원을 얻는다(J3은 A = J2, B = J4, J5는 A = J4, B = J6). 대상 관절의 영점이 맞으면 두 원은 겹치고, 틀리면 대상 축 둘레로 두 중심이 벌어진다. 보정량 δ$q$는 식 (2)로 구한다(식 (5)의 영점 오프셋과 δ$q$ = −Δ$q$ 관계).

$$ \delta q = -\arcsin\!\left(\frac{(c_B^{\perp}-c_A^{\perp})\cdot(a\times\hat v_A)}{\lVert v_B\rVert}\right) \qquad (2) $$

$c^{⊥}$는 대상 축 $a$에 수직인 평면으로 사영한 원 중심, $v_{A}$, $v_{B}$는 대상 관절 원점에서 각 중심까지의 벡터다. 오른팔 결과는 [Fig. 3]과 [Table 3]에 있다.

### 3.3 조건 ②: 직교 축의 방향 각

J5 축은 J6 축과 직교한다. J6과 J5 스윕 원의 법선을 마커 좌표로 옮기면, J6 영점이 맞을 때 J6 축 $n_{6}$ 둘레에서 본 J5 축의 방향은 설계 J5 축 $e_{5}^{nom}$과 같아야 한다. J6 보정량은 식 (3)으로 구한다.

$$ \delta q_6 = \operatorname{atan2}\!\big(\hat n_5^{\perp}\cdot(n_6\times e_5^{\mathrm{nom}}),\ \hat n_5^{\perp}\cdot e_5^{\mathrm{nom}}\big) + q_6^{\mathrm{enc}} \qquad (3) $$

$n̂_{5}^{⊥}$는 측정한 J5 축을 $n_{6}$에 수직인 평면에 사영한 단위 벡터, $q_{6}^{enc}$는 스윕 시작 J6 엔코더 각이다. J6 영점은 브라켓의 J6 동축 회전(yaw)과 구분되지 않으므로 브라켓 장착 정밀도가 한계다. D405 구성에서는 Step 2가 이 값에서 출발해 J6을 다시 푼다.

### 3.4 조건 ③: 원 반지름과 브라켓 위치

원 반지름은 축에서 마커까지의 거리와 같아야 한다. 마커 위치를 엔드이펙터 좌표 ($x$, $y$, $z$)로 두면 J6 축은 $z$, J5 축은 $y$ 방향이고, J5 축은 J6 축 방향으로 $L$만큼 떨어져 있어 엔코더 위상으로 피팅한 반지름은 식 (4)를 만족한다.

$$ r_6^2 = x^2 + y^2,\quad r_5^2 = x^2 + (z+L)^2 \qquad (4) $$

$y$는 설계값을 신뢰해 고정하고, $x$, $z$를 조립 공차 안에서 푼다. J4 반지름은 보정 전 J5 영점 오차를 그대로 받으므로 쓰지 않는다.

### 3.5 반복과 통과 기준

스윕, 피팅, 보정량 계산, 적용을 반복하며, [Table 2]의 기준을 모두 만족하면 통과로 보고 목표 근처 추정값의 평균을 보정값으로 쓴다. $s$는 추정값의 표본 표준편차이고, 잡음 하한 $s_{min}$은 우연히 가까운 두 값으로 통과하는 것을 막는다.

**[Table 2]** PASS CRITERIA OF STEP 1

| Item | Criterion |
|---|---|
| Iterations | ≤ 8 |
| Estimates used | Within 0.3° of target, {i:N} ≥ 2 |
| Standard error | max({i:s}, {i:s}{sub:min}) / √{i:N} ≤ 0.06° |
| Noise floor {i:s}{sub:min} | J5 0.03°, J6 0.06°, J3 0.12° |
| Step per iteration | ≤ 5.0° |

## 4. Step 1.5와 Step 2

### 4.1 관측 모델

같은 마커 자세를 로봇 경로(관절각 $q$와 영점 오프셋 Δ$q$의 순기구학)와 비전 경로(헤드 순기구학, 카메라 외부 파라미터, 카메라가 잰 마커 자세)로 계산하고, 자세 $k$의 잔차를 식 (5)로 정의한다. t5는 두 경로의 공통 기준인 토르소 최상단 링크다. Step 1.5는 이 잔차의 위치 성분을, Step 2는 6성분 전체를 쓴다.

$$ e_k(x) = \log\!\Big(\big(T_{t5,c}(x)\,T_{c,m}^{(k)}\big)^{-1}\,T_{t5,m}(q_k+\Delta q)\Big) \qquad (5) $$

### 4.2 Step 1.5: 헤드-카메라

양팔을 고정하고 헤드만 tilt ±10°, pan ±15°로 왕복하며, 식 (5)의 위치 성분으로 헤드 영점, 광축 회전, 카메라 위치를 범위 제약 최소제곱으로 푼다. 헤드에는 roll 관절이 없어 광축 회전 오차가 Step 2에서 양팔 J1의 공통 성분으로 흡수되므로(검증 자세 오차 1.00 → 0.24 mm) 이를 추정 변수에 넣었다.

### 4.3 Step 2: 앵커를 둔 전신 QP

식 (5)를 69자세에 대해 가중 최소제곱으로 풀고, 선형화한 정규방정식에 바운드를 붙여 OSQP[7]로 반복한다.

$$ \min_{\Delta x}\ \lVert J\Delta x + e\rVert_W^2 + \sum_i w_i\,(x_i + \Delta x_i - x_i^{\mathrm{ref}})^2 \qquad (6) $$

$W$는 자세 잔차 6성분(위치 3, 방향 3)의 측정 잡음 역분산이며 방향 표준편차 하한은 0.3°다. 둘째 항은 [Table 1]의 고정·앵커·댐핑 항으로, $x_{i}^{ref}$는 앞 단계 값, $w_{i}$는 가중이고 괄호 안 범위는 바운드다. Pass 1은 팔·헤드·카메라를 함께, Pass 2는 팔을 고정하고 헤드·카메라만 다시 푼다.

## 5. 실험 결과

[Fig. 3], [Fig. 4]와 [Table 3]이 보정 전후 비교다. 준비 자세 마커 오차는 같은 측정을 관절 오프셋 적용 전후의 모델 예측과 비교한 값이며(카메라·브라켓은 둘 다 보정값), 반복 실행을 뺀 리셋 6회의 첫 실행에서 양팔 값(n = 12)의 중앙값이다.

*(그림: fig2v12.png)*

**[Fig. 3]** Right elbow J3: circles of the J2 and J4 sweeps

*(그림: fig4.png)*

**[Fig. 4]** Check pose before and after: (a) front, (b) top view

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

Full Auto 소요 시간은 18분 32초였다. [Fig. 4]에서는 카메라를 거치지 않고도 두 브라켓 판이 보정 후 일직선으로 맞는 것이 보인다. [Table 4]는 같은 실행의 영점 오프셋으로 J1, J3이 1.4~2.4°로 가장 컸다(헤드 pan −0.07°, tilt −0.09°).

## 6. 결  론

RB-Y1의 16축 관절 영점을 원 기하, 헤드-카메라, 전신 QP의 세 단계로 나누어 보정하였다. Step 1의 기하 조건은 다른 7자유도 팔에도 적용할 수 있을 것이다. 다만 마커 오차는 외부 측정이 아니고 준비 자세는 Step 2 계획에 포함된 자세라 독립 검증이 아니므로, 향후 외부 측정기로 끝단 정확도를 평가할 예정이다.

## References

[1] R. Y. Tsai and R. K. Lenz, “A new technique for fully autonomous and efficient 3D robotics hand/eye calibration,” *IEEE Transactions on Robotics and Automation*, vol. 5, no. 3, pp. 345-358, 1989.
[2] F. C. Park and B. J. Martin, “Robot sensor calibration: solving AX = XB on the Euclidean group,” *IEEE Transactions on Robotics and Automation*, vol. 10, no. 5, pp. 717-721, 1994.
[3] J. M. Hollerbach and C. W. Wampler, “The calibration index and taxonomy for robot kinematic calibration methods,” *International Journal of Robotics Research*, vol. 15, no. 6, pp. 573-591, 1996.
[4] V. Pradeep, K. Konolige, and E. Berger, “Calibrating a multi-arm multi-sensor robot: A bundle adjustment approach,” *Experimental Robotics*, Springer Tracts in Advanced Robotics, vol. 79, Springer, 2014, pp. 211-225.
[5] D. Maier, S. Wrobel, and M. Bennewitz, “Whole-body self-calibration via graph-optimization and automatic configuration selection,” *IEEE International Conference on Robotics and Automation*, Seattle, USA, pp. 5662-5668, 2015.
[6] J. Tenhumberg, D. Winkelbauer, D. Burschka, and B. Bäuml, “Self-contained calibration of an elastic humanoid upper body using only a head-mounted RGB camera,” *IEEE-RAS International Conference on Humanoid Robots*, Ginowan, Japan, 2022.
[7] B. Stellato, G. Banjac, P. Goulart, A. Bemporad, and S. Boyd, “OSQP: An operator splitting solver for quadratic programs,” *Mathematical Programming Computation*, vol. 12, no. 4, pp. 637-672, 2020.
[8] J. Wang and E. Olson, “AprilTag 2: Efficient and robust fiducial detection,” *IEEE/RSJ International Conference on Intelligent Robots and Systems*, Daejeon, Korea, pp. 4193-4198, 2016.
[9] Z. Zhang, “A flexible new technique for camera calibration,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 22, no. 11, pp. 1330-1334, 2000.
