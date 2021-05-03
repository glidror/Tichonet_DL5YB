import numpy as np
import h5py
import os

def main():
    if __name__ == "__main__":
        main()

        
     
def load_dataB1W3Ex1():
    X = [15, 46, 45, 60,  67, 54, 73, 17, 68, 59, 12, 7, 31, 69, 92]
    Y = [34.7, 105.7, 80.6, 118.1, 136.8, 101.5, 146.0, 55.0, 119.2, 114.0, 48.7, 45.4, 69.2, 136.3, 185.9]
    return X, Y

def load_dataB1W3Ex2():
    X = [[  2.44545144,   6.14920217, -12.66906286,  14.61875342,
        14.7267701 ,  -2.70092236,   9.08391754,  10.88125055,
        11.74997654,  10.94831072, -12.47383907,   6.08392308,
         1.66746992, -11.19471334,   2.25364913,   1.88511582,
       -14.2274894 ,  11.84515153,   8.32367399,   2.97757991,
         2.64395449,  -7.36840356,  -7.61335084,  -8.91499837,
        10.82923428,  -4.58864561,  13.04620666, -14.19777802,
       -13.7707837 ,   8.18677396,  11.61767315, -12.25579077,
        -2.62260026,  -0.60676761,   3.32240745,   0.26801427,
        -4.18681283,  14.35415723,   6.53819795,   6.61630472,
        -6.27261278,  13.97740396,  -1.28014474,   0.82797197,
        -9.03235852,   9.26085051,  -0.64246221, -14.63801163,
       -12.71610996,  -2.41355138,   9.35879352,  10.94884679,
        12.84168121,  -7.05332504, -11.78455406,  -7.29804882,
        -9.59585489,   5.99550563,  -7.97315047,  -5.42703984,
         2.94665817, -14.00996681,   1.23154118,   7.65039358,
         6.04912491,  -2.16401687,  13.31856495,  -1.49517826,
        12.68551094, -12.37399579,   6.47768495,  11.93451094,
        -4.99806059, -13.603112  ,   0.05276469,  -3.20069084,
        -3.03461758,   6.43390155,  -6.28488378, -11.64834709,
        12.11489986,  -6.05724425,  -5.80750989,  -1.74298711,
         3.04478604,  -0.52814615,   6.34711324,   7.93562295,
        -4.09654101,  -5.69600533,   6.58399927,  -3.53451444,
       -10.55377674,   5.61397991,   6.86878068,  -4.25591381,
        10.72469616,  -3.09539764,  -5.72080877, -14.06657398], 
         [ 14.9660887 , -14.65572978,   5.87846727,   5.69524147,
        14.76476496,  -1.49337678,   6.29615159,  -7.84638361,
       -10.77133338,   8.58112   ,  12.62963093,   6.98238191,
        -3.75589664,  -2.63721403,  14.95188035,  13.37714696,
         5.07758005,   8.76318962,  12.87440822,   6.46890389,
         4.50888951,  -3.50326842,   1.14286491,   6.331831  ,
         1.51338619,   2.38652094,  -3.40969275, -10.30259633,
         5.58494709,  -7.48747586,   9.12020615,  -5.79435367,
        13.83689941,  -9.24642998,  -7.57452488,   4.53003823,
         3.74513147,  -9.04748393,   6.76415641,   8.37834971,
        12.53397601,  -0.38190918,  -7.65072391,   0.45443731,
        11.5260752 ,  -1.72819622,  -8.64443235,   5.56606852,
        10.96562565, -12.35320661,   8.21142224,  10.14268257,
       -13.06374421, -12.40626903,   7.98535131, -13.03048573,
        -4.73457543,  10.54999163,   3.60726855, -13.99742185,
        -5.89111013,  -8.15056781,  -6.48809874,  -6.23675839,
        13.47347069,   8.14863415, -12.11182034,  13.56307798,
        10.59410852,   6.95801561,   6.666816  , -14.83885578,
        10.64160557,  14.85290251,  14.56988349,  10.89957564,
       -11.45560215,  13.55520831,  -4.209692  ,  -8.44969244,
        -8.77033497,  10.41889229,   7.88699719,  11.61939483,
         8.71733151,  -1.15317556,   1.66230584, -14.02062789,
        14.00708275,  12.36866883,   0.16303983,  10.8067164 ,
        14.68737772,   0.03524456,   7.55187664,  -5.98074279,
        13.45613283,  -8.14969216,  -8.11398291, -13.88455254]]
    Y = [-1271.79437857,   269.57933963,  1185.4912226 ,  -456.99640165,
       -3107.6970408 ,   -43.87867319,  -560.94446708,   919.01696723,
        1687.38141057,  -993.23834845,  1511.06806957,  -565.58065144,
          22.44677629,   -13.57063298, -1166.44572213, -1047.91173983,
        1338.83166074, -1240.21246484, -1783.28941338,  -367.98964003,
        -221.9831084 ,  -175.79708346,   286.33988907,   757.87745596,
         114.29940897,   148.80772041,  1056.08535927, -1377.18851497,
        1215.58688911,   614.17739403, -1063.4809326 ,  -530.53097368,
        -264.66802547,  -408.49254454,    91.9767231 ,   -81.86294907,
         187.24615356,  1955.8074885 ,  -534.30229188,  -624.43669311,
         469.71886444,   688.22274266,  -357.02169011,   -10.82971626,
        1050.92083791,   390.42031338,  -344.14530974,  1385.82626988,
        1614.68155171,  -997.32214464,  -754.77576723, -1267.74733681,
        1669.98130027, -1523.51882634,  1190.19350675, -1570.45409898,
        -397.31482709,  -987.02152334,   407.84495252, -1420.58406075,
          92.30135236,  -856.55787308,   -66.30979002,   572.23195692,
       -1376.15017148,   -45.56750571,  1969.27899915,  -518.42758689,
       -1609.48208354,  1370.38779242,  -521.8184776 ,  1311.63582744,
         225.87395052,  2503.04254946,  -754.88834945,   -32.54890635,
        -804.11527715, -1252.06810521,  -286.5193541 ,  -971.12029928,
        1225.27614243,   385.05608341,   380.76934103,  -273.76867277,
        -608.43417205,   -20.59537018,   -24.71129014,   632.13398344,
         -51.26253441,   308.63882654,    97.20334158,    21.56552297,
        1494.74139927,    82.01790897,  -676.65024283,  -356.35960896,
       -2076.27928591,  -555.42266107,  -654.76462096, -2144.2232722 ]
    return X, Y

def load_dataB1W4_trainN():
    X = np.array( [[15,	1800000,	5800,	50],
        [15, 	1790000,	6200,	50],
        [15, 	1780000,	6400,	60],
        [25,  	1778000,	6500,	60],
        [25, 	1750000,	6550,	60],
        [25, 	1740000,	6580,	70],
        [25, 	1725000,	8200,	75],
        [30, 	1725000,	8600,	75],
        [30, 	1720000,	8800,	75],
        [30, 	1705000,	9200,	80],
        [30, 	1710000,	9630,	80],
        [40, 	1700000,	10570,	80],
        [40 ,	1695000,	11330,	85],
        [40, 	1695000,	11600,	100],
        [40 ,	1690000,	11800,	105],
        [40, 	1630000,	11830,	105],
        [65 ,	1640000,	12650,	105],
        [102, 	1635000,	13000,	110],
        [75 ,	1630000,	13224,	125],
        [75 ,	1620000,	13766,	130],
        [75 ,	1615000	,14010,	150],
        [80, 	1605000	,14468,	155],
        [86 ,	1590000	,15000,	165],
        [98 ,	1595000,	15200,	175],
        [87 ,	1590000,	15600,	175],
        [77 ,	1600000,	16000,	190],
        [63 ,	1610000,	16200,	200]])
    Y = np.array([192000, 190400,191200,177600,176800,178400,180800,175200,
        174400,173920,172800,163200,161600,161600,160800,159200,
        148800,115696,147200,150400,152000,136000,126240,123888,
        126080,151680,152800])
    return X.T, Y

