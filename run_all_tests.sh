
# Table 1 (SOTA comparison)
echo "\033[1;36mTable 1 (SOTA comparison)\033[0m"
## Yin et al 
bash run_depth_test_eval.sh config/depth_exp1_bsl.cfg
echo "\033[1;36m    0.1546,     1.2965,     5.8574,     0.2332,     0.0000,     0.7931,     0.9314,     0.9725\033[0m\n"

## Ours
bash run_depth_test_eval.sh config/depth_exp1_ours.cfg
echo "\033[1;36m    0.1329,     0.9051,     5.1809,     0.2079,     0.0000,     0.8245,     0.9467,     0.9805\033[0m\n"


# Table 2 (semantic ablation)
echo "\n\n\033[1;36mTable 2 (semantic ablation)\033[0m"
## Dense semantic

bash run_depth_test_eval.sh config/depth_exp2_row1.cfg
echo "\033[1;36m    0.1418,     0.9914,     5.3089,     0.2156,     0.0000,     0.8144,     0.9425,     0.9779\033[0m\n"
echo "\033[0;33m    This slight difference is because we re-ran the training under this configuration \033[0m\n"

## One-hot semantic
bash run_depth_test_eval.sh config/depth_exp2_row2.cfg
echo "\033[1;36m    0.1390,     0.9487,     5.2266,     0.2135,     0.0000,     0.8177,     0.9450,     0.9795\033[0m\n"

## Dense instance
bash run_depth_test_eval.sh config/depth_exp2_row3.cfg
echo "\033[1;36m    0.1422,     0.9862,     5.3254,     0.2183,     0.0000,     0.8124,     0.9425,     0.9776\033[0m\n"

## One-hot instance
bash run_depth_test_eval.sh config/depth_exp2_row4.cfg
echo "\033[1;36m    0.1406,     0.9757,     5.2718,     0.2153,     0.0000,     0.8105,     0.9417,     0.9786\033[0m\n"

## Instance edge
bash run_depth_test_eval.sh config/depth_exp2_row5.cfg
echo "\033[1;36m    0.1453,     1.0367,     5.3144,     0.2168,     0.0000,     0.8069,     0.9425,     0.9784\033[0m\n"

## Dense instance + edge
bash run_depth_test_eval.sh config/depth_exp2_row6.cfg
echo "\033[1;36m    0.1416,     0.9693,     5.4469,     0.2194,     0.0000,     0.8080,     0.9412,     0.9777\033[0m\n"

## One hot semantic+ instance + edge
bash run_depth_test_eval.sh config/depth_exp2_row7.cfg
echo "\033[1;36m    0.1329,     0.9051,     5.1809,     0.2079,     0.0000,     0.8245,     0.9467,     0.9805\033[0m\n"


# Table 3 (arch ablation)
echo "\n\n\033[1;36mTable 3 (arch ablation)\033[0m"
## DepthNet channel
bash run_depth_test_eval.sh config/depth_exp3_row1.cfg
echo "\033[1;36m    0.1445,     0.9574,     5.2913,     0.2163,     0.0000,     0.8048,     0.9428,     0.9795\033[0m\n"

## PoseNet channel
bash run_depth_test_eval.sh config/depth_exp3_row2.cfg
echo "\033[1;36m    0.1472,     1.0757,     5.3852,     0.2225,     0.0000,     0.8084,     0.9383,     0.9752\033[0m\n"

## DepthNet + PoseNet channel
bash run_depth_test_eval.sh config/depth_exp3_row3.cfg
echo "\033[1;36m    0.1390,     0.9487,     5.2266,     0.2135,     0.0000,     0.8177,     0.9450,     0.9795\033[0m\n"

## Extra DepthNet + PoseNet channel
bash run_depth_test_eval.sh config/depth_exp3_row4.cfg
echo "\033[1;36m    0.1467,     1.0356,     5.5926,     0.2256,     0.0000,     0.8027,     0.9366,     0.9751\033[0m\n"

## DepthNet channel + Extra PoseNet
bash run_depth_test_eval.sh config/depth_exp3_row5.cfg
echo "\033[1;36m    0.1353,     0.9323,     5.2412,     0.2105,     0.0000,     0.8206,     0.9451,     0.9797\033[0m\n"


# Table 4 (transfer network)
echo "\n\n\033[1;36mTable 4 (transfer network)\033[0m"
## Transfer
bash run_depth_test_eval.sh config/depth_exp4_row1.cfg
echo "\033[1;36m    0.1502,     1.1414,     5.7088,     0.2305,     0.0000,     0.7922,     0.9343,     0.9739\033[0m\n"

## Transfer + scale normalization
bash run_depth_test_eval.sh config/depth_exp4_row3.cfg
echo "\033[1;36m    0.1445,     0.9944,     5.4222,     0.2217,     0.0000,     0.8061,     0.9386,     0.9763\033[0m\n"