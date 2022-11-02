\ Model Avocado_Price_Allocation
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 0.3 x[0] - 0.1 x[1] - 0.4 x[2] - 0.5 x[3] - 0.3 x[4] - 0.2 x[5]
   - 0.2 x[6] - 0.2 x[7] - 0.1 w[0] - 0.1 w[1] - 0.1 w[2] - 0.1 w[3]
   - 0.1 w[4] - 0.1 w[5] - 0.1 w[6] - 0.1 w[7] + [
   2 s[0] * reg_features[0,8] + 2 s[1] * reg_features[1,8]
   + 2 s[2] * reg_features[2,8] + 2 s[3] * reg_features[3,8]
   + 2 s[4] * reg_features[4,8] + 2 s[5] * reg_features[5,8]
   + 2 s[6] * reg_features[6,8] + 2 s[7] * reg_features[7,8] ] / 2
Subject To
 R0: x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] = 30
 R1: - x[0] + s[0] <= 0
 R2: - x[1] + s[1] <= 0
 R3: - x[2] + s[2] <= 0
 R4: - x[3] + s[3] <= 0
 R5: - x[4] + s[4] <= 0
 R6: - x[5] + s[5] <= 0
 R7: - x[6] + s[6] <= 0
 R8: - x[7] + s[7] <= 0
 R9: s[0] - demand[0] <= 0
 R10: s[1] - demand[1] <= 0
 R11: s[2] - demand[2] <= 0
 R12: s[3] - demand[3] <= 0
 R13: s[4] - demand[4] <= 0
 R14: s[5] - demand[5] <= 0
 R15: s[6] - demand[6] <= 0
 R16: s[7] - demand[7] <= 0
 R17: - x[0] + s[0] + w[0] = 0
 R18: - x[1] + s[1] + w[1] = 0
 R19: - x[2] + s[2] + w[2] = 0
 R20: - x[3] + s[3] + w[3] = 0
 R21: - x[4] + s[4] + w[4] = 0
 R22: - x[5] + s[5] + w[5] = 0
 R23: - x[6] + s[6] + w[6] = 0
 R24: - x[7] + s[7] + w[7] = 0
 LinearRegressionConstr1.linreg[0,0]: 0.2426931536778747 reg_features[0,0]
   - 1.433019836643237 reg_features[0,1]
   + 3.01924379063788 reg_features[0,2]
   + 1.815032872390411 reg_features[0,3]
   - 1.713812073973461 reg_features[0,4]
   - 0.5839901666611677 reg_features[0,5]
   - 2.601799448641272 reg_features[0,6]
   - 0.1607693023184433 reg_features[0,7]
   + 2.203770104890248 reg_features[0,8]
   - 0.5485105058308409 reg_features[0,9] + demand[0] = 5.439310052165032
 LinearRegressionConstr1.linreg[1,0]: 0.2426931536778747 reg_features[1,0]
   - 1.433019836643237 reg_features[1,1]
   + 3.01924379063788 reg_features[1,2]
   + 1.815032872390411 reg_features[1,3]
   - 1.713812073973461 reg_features[1,4]
   - 0.5839901666611677 reg_features[1,5]
   - 2.601799448641272 reg_features[1,6]
   - 0.1607693023184433 reg_features[1,7]
   + 2.203770104890248 reg_features[1,8]
   - 0.5485105058308409 reg_features[1,9] + demand[1] = 5.439310052165032
 LinearRegressionConstr1.linreg[2,0]: 0.2426931536778747 reg_features[2,0]
   - 1.433019836643237 reg_features[2,1]
   + 3.01924379063788 reg_features[2,2]
   + 1.815032872390411 reg_features[2,3]
   - 1.713812073973461 reg_features[2,4]
   - 0.5839901666611677 reg_features[2,5]
   - 2.601799448641272 reg_features[2,6]
   - 0.1607693023184433 reg_features[2,7]
   + 2.203770104890248 reg_features[2,8]
   - 0.5485105058308409 reg_features[2,9] + demand[2] = 5.439310052165032
 LinearRegressionConstr1.linreg[3,0]: 0.2426931536778747 reg_features[3,0]
   - 1.433019836643237 reg_features[3,1]
   + 3.01924379063788 reg_features[3,2]
   + 1.815032872390411 reg_features[3,3]
   - 1.713812073973461 reg_features[3,4]
   - 0.5839901666611677 reg_features[3,5]
   - 2.601799448641272 reg_features[3,6]
   - 0.1607693023184433 reg_features[3,7]
   + 2.203770104890248 reg_features[3,8]
   - 0.5485105058308409 reg_features[3,9] + demand[3] = 5.439310052165032
 LinearRegressionConstr1.linreg[4,0]: 0.2426931536778747 reg_features[4,0]
   - 1.433019836643237 reg_features[4,1]
   + 3.01924379063788 reg_features[4,2]
   + 1.815032872390411 reg_features[4,3]
   - 1.713812073973461 reg_features[4,4]
   - 0.5839901666611677 reg_features[4,5]
   - 2.601799448641272 reg_features[4,6]
   - 0.1607693023184433 reg_features[4,7]
   + 2.203770104890248 reg_features[4,8]
   - 0.5485105058308409 reg_features[4,9] + demand[4] = 5.439310052165032
 LinearRegressionConstr1.linreg[5,0]: 0.2426931536778747 reg_features[5,0]
   - 1.433019836643237 reg_features[5,1]
   + 3.01924379063788 reg_features[5,2]
   + 1.815032872390411 reg_features[5,3]
   - 1.713812073973461 reg_features[5,4]
   - 0.5839901666611677 reg_features[5,5]
   - 2.601799448641272 reg_features[5,6]
   - 0.1607693023184433 reg_features[5,7]
   + 2.203770104890248 reg_features[5,8]
   - 0.5485105058308409 reg_features[5,9] + demand[5] = 5.439310052165032
 LinearRegressionConstr1.linreg[6,0]: 0.2426931536778747 reg_features[6,0]
   - 1.433019836643237 reg_features[6,1]
   + 3.01924379063788 reg_features[6,2]
   + 1.815032872390411 reg_features[6,3]
   - 1.713812073973461 reg_features[6,4]
   - 0.5839901666611677 reg_features[6,5]
   - 2.601799448641272 reg_features[6,6]
   - 0.1607693023184433 reg_features[6,7]
   + 2.203770104890248 reg_features[6,8]
   - 0.5485105058308409 reg_features[6,9] + demand[6] = 5.439310052165032
 LinearRegressionConstr1.linreg[7,0]: 0.2426931536778747 reg_features[7,0]
   - 1.433019836643237 reg_features[7,1]
   + 3.01924379063788 reg_features[7,2]
   + 1.815032872390411 reg_features[7,3]
   - 1.713812073973461 reg_features[7,4]
   - 0.5839901666611677 reg_features[7,5]
   - 2.601799448641272 reg_features[7,6]
   - 0.1607693023184433 reg_features[7,7]
   + 2.203770104890248 reg_features[7,8]
   - 0.5485105058308409 reg_features[7,9] + demand[7] = 5.439310052165032
Bounds
 2.06357351 <= x[0] <= 7.094764730000001
 1.84544313 <= x[1] <= 6.168571610000001
 2.36442449 <= x[2] <= 8.836406220000001
 0.2196899 <= x[3] <= 0.91798395
 3.68713018 <= x[4] <= 10.32317459
 2.1977637 <= x[5] <= 7.81047462
 3.26010217 <= x[6] <= 11.27474911
 1.05893809 <= x[7] <= 3.57549921
 reg_features[0,0] = 0
 reg_features[0,1] = 0
 reg_features[0,2] = 0
 reg_features[0,3] = 0
 reg_features[0,4] = 0
 reg_features[0,5] = 0
 reg_features[0,6] = 0
 reg_features[0,7] = 7
 reg_features[0,8] <= 2
 reg_features[0,9] = 1
 reg_features[1,0] = 1
 reg_features[1,1] = 0
 reg_features[1,2] = 0
 reg_features[1,3] = 0
 reg_features[1,4] = 0
 reg_features[1,5] = 0
 reg_features[1,6] = 0
 reg_features[1,7] = 7
 reg_features[1,8] <= 2
 reg_features[1,9] = 1
 reg_features[2,0] = 0
 reg_features[2,1] = 1
 reg_features[2,2] = 0
 reg_features[2,3] = 0
 reg_features[2,4] = 0
 reg_features[2,5] = 0
 reg_features[2,6] = 0
 reg_features[2,7] = 7
 reg_features[2,8] <= 2
 reg_features[2,9] = 1
 reg_features[3,0] = 0
 reg_features[3,1] = 0
 reg_features[3,2] = 1
 reg_features[3,3] = 0
 reg_features[3,4] = 0
 reg_features[3,5] = 0
 reg_features[3,6] = 0
 reg_features[3,7] = 7
 reg_features[3,8] <= 2
 reg_features[3,9] = 1
 reg_features[4,0] = 0
 reg_features[4,1] = 0
 reg_features[4,2] = 0
 reg_features[4,3] = 0
 reg_features[4,4] = 1
 reg_features[4,5] = 0
 reg_features[4,6] = 0
 reg_features[4,7] = 7
 reg_features[4,8] <= 2
 reg_features[4,9] = 1
 reg_features[5,0] = 0
 reg_features[5,1] = 0
 reg_features[5,2] = 0
 reg_features[5,3] = 0
 reg_features[5,4] = 0
 reg_features[5,5] = 1
 reg_features[5,6] = 0
 reg_features[5,7] = 7
 reg_features[5,8] <= 2
 reg_features[5,9] = 1
 reg_features[6,0] = 0
 reg_features[6,1] = 0
 reg_features[6,2] = 0
 reg_features[6,3] = 0
 reg_features[6,4] = 0
 reg_features[6,5] = 0
 reg_features[6,6] = 1
 reg_features[6,7] = 7
 reg_features[6,8] <= 2
 reg_features[6,9] = 1
 reg_features[7,0] = 0
 reg_features[7,1] = 0
 reg_features[7,2] = 0
 reg_features[7,3] = 1
 reg_features[7,4] = 0
 reg_features[7,5] = 0
 reg_features[7,6] = 0
 reg_features[7,7] = 7
 reg_features[7,8] <= 2
 reg_features[7,9] = 1
 demand[0] free
 demand[1] free
 demand[2] free
 demand[3] free
 demand[4] free
 demand[5] free
 demand[6] free
 demand[7] free
End
