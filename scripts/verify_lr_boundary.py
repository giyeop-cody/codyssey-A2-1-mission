"""학습률 발산 조건을 함수별로 재실행하여 검증한다."""
import pathlib,json
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
rows=[]
for name,hess in [('x²+y²',np.diag([2.,2.])),('x²+10y²',np.diag([2.,20.]))]:
 p=np.array([3.,2.]); history=[p.tolist()]
 for _ in range(12): p=p-.5*(hess@p);history.append(p.tolist())
 rows.append({'function':name,'lr':.5,'iterations':12,'hessian_max':float(np.linalg.eigvalsh(hess)[-1]),
              'stable_lr_strict_upper':float(2/np.linalg.eigvalsh(hess)[-1]),'path':history,
              'final_radius':float(np.linalg.norm(p))})
assert rows[0]['final_radius']==0 and rows[1]['final_radius']>1e8
for lr in [.9,1.,1.05,1.5]:
 p=np.array([3.,2.]);history=[p.tolist()]
 for _ in range(12):p=p-lr*2*p;history.append(p.tolist())
 rows.append(dict(function='x²+y²',lr=lr,iterations=12,path=history,final_radius=float(np.linalg.norm(p))))
assert rows[2]['final_radius']<np.linalg.norm([3.,2.])
assert np.isclose(rows[3]['final_radius'],np.linalg.norm([3.,2.]))
assert rows[4]['final_radius']>np.linalg.norm([3.,2.])
assert rows[5]['final_radius']>1e4
(ROOT/'reports/lr-boundary-verification.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print(rows[0]['final_radius'],rows[1]['final_radius'])
