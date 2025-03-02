"""
Tentative train/valid/test split for different tasks
"""

# List: [ID]
odom_sample = [["boreas-2021-09-02-11-42"]]

odom_train = [
    ["boreas-2020-11-26-13-58"],
    ["boreas-2020-12-01-13-26"],
    ["boreas-2020-12-18-13-44"],
    ["boreas-2021-01-15-12-17"],
    ["boreas-2021-01-19-15-08"],
    ["boreas-2021-01-26-11-22"],
    ["boreas-2021-02-02-14-07"],
    ["boreas-2021-03-02-13-38"],
    ["boreas-2021-03-23-12-43"],
    ["boreas-2021-03-30-14-23"],
    ["boreas-2021-04-08-12-44"],
    ["boreas-2021-04-13-14-49"],
    ["boreas-2021-04-15-18-55"],
    ["boreas-2021-04-20-14-11"],
    ["boreas-2021-04-22-15-00"],
    ["boreas-2021-04-29-15-55"],
    ["boreas-2021-05-06-13-19"],
    ["boreas-2021-05-13-16-11"],
    ["boreas-2021-06-03-16-00"],
    ["boreas-2021-06-17-17-52"],
    ["boreas-2021-07-20-17-33"],
    ["boreas-2021-07-27-14-43"],
    ["boreas-2021-08-05-13-34"],
    ["boreas-2021-09-02-11-42"],
    ["boreas-2021-09-07-09-35"],
    ["boreas-2021-09-14-20-00"],
    ["boreas-2021-10-15-12-35"],
    ["boreas-2021-10-22-11-36"],
    ["boreas-2021-11-02-11-16"],
    ["boreas-2021-11-14-09-47"],
    ["boreas-2021-11-16-14-10"],
    ["boreas-2021-11-23-14-27"],
]

odom_test = [
    ["boreas-2020-12-04-14-00"],
    ["boreas-2021-01-26-10-59"],
    ["boreas-2021-02-09-12-55"],
    ["boreas-2021-03-09-14-23"],
    ["boreas-2021-04-22-15-00"],
    ["boreas-2021-06-29-18-53"],
    ["boreas-2021-06-29-20-43"],
    ["boreas-2021-09-08-21-00"],
    ["boreas-2021-09-09-15-28"],
    ["boreas-2021-10-05-15-35"],
    ["boreas-2021-10-26-12-35"],
    ["boreas-2021-11-06-18-55"],
    ["boreas-2021-11-28-09-18"],
]

loc_sample = odom_sample
loc_train = [
    ["boreas-2020-11-26-13-58"],
    ["boreas-2020-12-01-13-26"],
    ["boreas-2020-12-18-13-44"],
    ["boreas-2021-01-15-12-17"],
    ["boreas-2021-01-19-15-08"],
    ["boreas-2021-01-26-11-22"],
    ["boreas-2021-02-02-14-07"],
    ["boreas-2021-03-02-13-38"],
    ["boreas-2021-03-23-12-43"],
    ["boreas-2021-03-30-14-23"],
    ["boreas-2021-04-08-12-44"],
    ["boreas-2021-04-13-14-49"],
    ["boreas-2021-04-15-18-55"],
    ["boreas-2021-04-20-14-11"],
    ["boreas-2021-04-20-14-11"],
    ["boreas-2021-04-29-15-55"],
    ["boreas-2021-05-06-13-19"],
    ["boreas-2021-05-13-16-11"],
    ["boreas-2021-06-03-16-00"],
    ["boreas-2021-06-17-17-52"],
    ["boreas-2021-08-05-13-34"],
    ["boreas-2021-09-02-11-42"],
    ["boreas-2021-09-07-09-35"],
    ["boreas-2021-10-15-12-35"],
    ["boreas-2021-10-22-11-36"],
    ["boreas-2021-11-02-11-16"],
    ["boreas-2021-11-14-09-47"],
    ["boreas-2021-11-16-14-10"],
    ["boreas-2021-11-23-14-27"],
]

loc_test = [
    ["boreas-2020-12-04-14-00"],
    ["boreas-2021-01-26-10-59"],
    ["boreas-2021-02-09-12-55"],
    ["boreas-2021-03-09-14-23"],
    ["boreas-2021-06-29-18-53"],
    ["boreas-2021-09-08-21-00"],
    ["boreas-2021-10-05-15-35"],
    ["boreas-2021-10-26-12-35"],
    ["boreas-2021-11-06-18-55"],
    ["boreas-2021-11-28-09-18"],
]

loc_reference = "boreas-2020-11-26-13-58"

obj_sample = [["boreas-objects-v1", 1598990052112791, 1598990071610029]]

obj_train = [
    ["boreas-objects-v1", 1598989992068558, 1598990031683161],
    ["boreas-objects-v1", 1598988939059928, 1598988958763542],
    ["boreas-objects-v1", 1598990950088125, 1598990989703096],
    ["boreas-objects-v1", 1598989516069069, 1598989574765082],
    ["boreas-objects-v1", 1598992245041538, 1598992264745062],
    ["boreas-objects-v1", 1598987772079070, 1598987827664223],
    ["boreas-objects-v1", 1598987327084848, 1598987346789002],
    ["boreas-objects-v1", 1598987854109010, 1598987893723823],
    ["boreas-objects-v1", 1598989321106021, 1598989392972662],
    ["boreas-objects-v1", 1598989897075981, 1598989916779960],
    ["boreas-objects-v1", 1598988970067337, 1598988989771314],
    ["boreas-objects-v1", 1598987513026635, 1598987532730090],
    ["boreas-objects-v1", 1598986289111738, 1598986308607975],
    ["boreas-objects-v1", 1598991443097712, 1598991462801660],
    ["boreas-objects-v1", 1598989109031027, 1598989158809311],
    ["boreas-objects-v1", 1598989393180743, 1598989412676672],
    ["boreas-objects-v1", 1598990189106857, 1598990208603736],
    ["boreas-objects-v1", 1598989277032742, 1598989296736234],
    ["boreas-objects-v1", 1598991312015344, 1598991331718988],
    ["boreas-objects-v1", 1598986334015318, 1598986353719546],
    ["boreas-objects-v1", 1598988259072791, 1598988314969159],
    ["boreas-objects-v1", 1598986432015738, 1598986451719325],
    ["boreas-objects-v1", 1598990635035777, 1598990654739378],
    ["boreas-objects-v1", 1598989775016310, 1598989794720263],
    ["boreas-objects-v1", 1598988365161600, 1598988384657794],
    ["boreas-objects-v1", 1598991975100867, 1598991994804056],
    ["boreas-objects-v1", 1598987396048380, 1598987415752230],
    ["boreas-objects-v1", 1598988315176496, 1598988364954129],
    ["boreas-objects-v1", 1598988066079643, 1598988125606182],
    ["boreas-objects-v1", 1598992517056146, 1598992536760342],
    ["boreas-objects-v1", 1598988199132614, 1598988258865374],
    ["boreas-objects-v1", 1598987162091512, 1598987181796248],
    ["boreas-objects-v1", 1598990841095344, 1598990860799051],
    ["boreas-objects-v1", 1598987710997913, 1598987728627278],
    ["boreas-objects-v1", 1598989159016617, 1598989204750120],
    ["boreas-objects-v1", 1598992458049273, 1598992477752741],
    ["boreas-objects-v1", 1598990052112791, 1598990071610029],
]

obj_test = [
    ["boreas-objects-v1", 1598988139087912, 1598988198925188],
    ["boreas-objects-v1", 1598990697051024, 1598990716754824],
    ["boreas-objects-v1", 1598991652062063, 1598991671765636],
    ["boreas-objects-v1", 1598987586034628, 1598987641620061],
    ["boreas-objects-v1", 1598990889110342, 1598990908606355],
    ["boreas-objects-v1", 1598986831068188, 1598986850771570],
    ["boreas-objects-v1", 1598988013087767, 1598988046687950],
    ["boreas-objects-v1", 1598992318048815, 1598992337752918],
    ["boreas-objects-v1", 1598986676031073, 1598986695734963],
    ["boreas-objects-v1", 1598989467019312, 1598989506632468],
    ["boreas-objects-v1", 1598987654064765, 1598987710790471],
    ["boreas-objects-v1", 1598987228046683, 1598987247750754],
    ["boreas-objects-v1", 1598989052097452, 1598989091712377],
    ["boreas-objects-v1", 1598987071038383, 1598987090742468],
    ["boreas-objects-v1", 1598989635016559, 1598989654720273],
    ["boreas-objects-v1", 1598986746031268, 1598986765734397],
]