import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-submit_file")
parser.add_argument("-predict_file")
args = parser.parse_args()

f1 = open(args.predict_file)
all_data = f1.readlines()

rs_all_data1 = []
rs_str = ''
pre_tag = ''
rs_all_data = []
idx=0
all_data.append('\n')
for str in all_data:
    str = str.strip()
    if len(str)==0:
        idx=idx+1
        if len(rs_str)>0:
            if rs_str.startswith('_'):
                rs_str = rs_str[1:]
            if rs_str.endswith('_'):
                rs_str = rs_str[:-2]
            rs_str = rs_str + '/' + pre_tag
            rs_all_data.append(rs_str)
            rs_str = ''
            pre_tag = ''

            rs_all_data1.append(rs_all_data)
            rs_all_data=[]
        else:
            if len(rs_all_data)>0:
                rs_all_data1.append(rs_all_data)
                rs_all_data = []
    else:
        sss = str.split("\t")

        if len(rs_str)==0:
            if '_' in sss[2]:
                sss[2] = sss[2].split("_")[1]
            rs_str  = rs_str+'_'+sss[0]
            pre_tag = sss[2]
        else:
            if sss[2].startswith('B'):
                if '_' in sss[2]:
                    sss[2] = sss[2].split("_")[1]
                if len(rs_str) == 0:
                    rs_str = rs_str + '_' + sss[0]
                    pre_tag = sss[2]
                else:
                    if rs_str.startswith('_'):
                        rs_str = rs_str[1:]
                    if rs_str.endswith('_'):
                        rs_str = rs_str[:-2]
                    rs_str = rs_str + '/' + pre_tag
                    rs_all_data.append(rs_str)
                    rs_str = ''
                    rs_str = rs_str + '_' + sss[0]
                    pre_tag = sss[2]
            else:
                if '_' in sss[2]:
                    sss[2] = sss[2].split("_")[1]
                if pre_tag == sss[2]:
                    rs_str = rs_str + '_' + sss[0]
                else:
                    if rs_str.startswith('_'):
                        rs_str = rs_str[1:]
                    if rs_str.endswith('_'):
                        rs_str = rs_str[:-2]
                    rs_str = rs_str + '/' + pre_tag
                    rs_all_data.append(rs_str)
                    rs_str = ''
                    rs_str = rs_str + '_' + sss[0]
                    pre_tag = sss[2]
                    
print(idx)
print(len(rs_all_data1))
f2 = open(args.submit_file,'w')
for ss in rs_all_data1:
    f2.write('  '.join(ss).lower() + "\n")
f2.close()