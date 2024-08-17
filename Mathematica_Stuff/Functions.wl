(* ::Package:: *)

(* Define the Baseline function *)
Baseline[f_, U_, Cknob_, eta_, trim_, Cstray_, phiConst_, 
  DCOffset_] := 
 Module[{circConsts, pi, imUnit, sign, L0, Rcoil, R, R1, r, alpha, 
   beta1, ZCable, D, M, deltaC, deltaPhi, deltaPhase, deltaL, 
   I, wRes, wLow, wHigh, deltaW, w, slope, slopePhi, Ctrim, Cmain, 
   C, Z0, beta, gamma, ZC, vel, l, ic, chi, pt, L, ZLpure, Zstray, 
   ZL, ZT, Zleg1, Ztotal, parfaze, phiTrim, phi, VOut, outY, offset},
  
  (* Preamble *)
  circConsts = {3*10^-8, 0.35, 619, 50, 10, 0.0343, 4.752*10^-9, 50, 
    1.027*10^-10, 2.542*10^-7, 0, 0, 0, 0};
  {L0, Rcoil, R, R1, r, alpha, beta1, ZCable, D, M, deltaC, deltaPhi, 
   deltaPhase, deltaL} = circConsts;
  pi = Pi;
  imUnit = I;
  sign = 1;
  
  (* Main constants *)
  I = U*1000/R;
  wRes = 2*pi*213*10^6;
  wLow = 2*pi*(213 - 4)*10^6;
  wHigh = 2*pi*(213 + 4)*10^6;
  deltaW = 2*pi*4*10^6/500;
  
  (* Convert frequency to angular frequency *)
  w = 2*pi*f*10^6;
  
  (* Functions *)
  slope = deltaC/(0.25*2*pi*10^6);
  slopePhi = deltaPhi/(0.25*2*pi*10^6);
  Ctrim[w_] := slope*(w - wRes);
  Cmain[] := 20*10^-12*Cknob;
  C[w_] := Cmain[] + Ctrim[w]*10^-12;
  
  Z0[w_] := 
   Module[{S = 2*ZCable*alpha}, 
    If[w == 0, 0, 
     Sqrt[(S + w*M*imUnit)/(w*D*imUnit)]]];
  beta[w_] := beta1*w;
  gamma[w_] := alpha + beta[w]*I;
  ZC[w_] := If[C[w] != 0, 1/(imUnit*w*C[w]), 0];
  vel[w_] := 1/beta[w];
  l[w_] := trim*vel[wRes] + deltaL;
  ic[w_] := 0.11133;
  chi[w_] := ConstantArray[0, Length[w]]; (* Placeholder *)
  pt[w_] := ic[w];
  L[w_] := L0*(1 + sign*4*pi*eta*pt[w]*chi[w]);
  ZLpure[w_] := imUnit*w*L[w] + Rcoil;
  Zstray[w_] := If[Cstray != 0, 1/(imUnit*w*Cstray), 0];
  ZL[w_] := ZLpure[w]*Zstray[w]/(ZLpure[w] + Zstray[w]);
  
  ZT[w_] := 
   Module[{epsilon = 10^-10}, 
    Z0[w]*(ZL[w] + Z0[w]*Tanh[gamma[w]*l[w]])/
     (Z0[w] + ZL[w]*Tanh[gamma[w]*l[w]] + epsilon)];
  
  Zleg1[w_] := r + ZC[w] + ZT[w];
  Ztotal[w_] := R1/(1 + (R1/Zleg1[w]));
  
  parfaze[w_] := 
   Module[{a, bb, c, xp1 = wLow, xp2 = wRes, xp3 = wHigh, yp1 = 0, 
     yp2 = deltaPhase, yp3 = 0},
    a = ((yp1 - yp2)*(wLow - wHigh) - (yp1 - yp3)*(wLow - wRes))/
      (((wLow^2) - (wRes^2))*(wLow - wHigh) - 
       ((wLow^2) - (wHigh^2))*(wLow - wRes));
    bb = (yp1 - yp3 - a*((wLow^2) - (wHigh^2)))/(wLow - wHigh);
    c = yp1 - a*(wLow^2) - bb*wLow;
    a*w^2 + bb*w + c];
  
  phiTrim[w_] := slopePhi*(w - wRes) + parfaze[w];
  phi[w_] := phiTrim[w] + phiConst;
  
  VOut[w_] := -1*(I*Ztotal[w]*Exp[imUnit*phi[w]*pi/180]);
  
  outY = VOut[w];
  offset = Map[# - Min[Re[outY]] &, Re[outY]];
  offset + DCOffset
  ]
