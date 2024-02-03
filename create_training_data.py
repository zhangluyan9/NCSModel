import numpy as np
import pandas as pd
import torch
import numpy as np
from gensim.models import Word2Vec
import torch
import gensim
import math

def positional_encoding(seq_len, d_model):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

    pos_enc = torch.zeros(seq_len, d_model)
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    pos_enc[:, 1::2] = torch.cos(position * div_term)

    return pos_enc

data_str = """
floureint	STD	Genèses
35837	337.0	124111111123131131113411311222112411324122311
23510	174.0	124411124124122111111141122111311423111111443
23046	324.3	124141122113313112122431434322113131324113342
22952	348.7	124222124111232132442122421342111422412411221
21946	142.4	124112121112422412424114311113414112412121311
21802	289.4	124113212113123141441411112122411424131333434
20912	769.4	124431114213132141411412123411111221421111411
20798	481.6	124421111421212212113442412123111414113421224
20088	331.2	124311113442111421111244223113113411111441223
17889	318.8	124411134112411311131124334134111212412334431
17256	399.5	124432111411111223413342233111231312433112122
15325	526.4	124122134111132143111112432432342322111141313
14258	335.1	124231411314111111423421221431213231441442322
13613	147.4	124141113411343141111114111111132221221324131
13603	253.7	124142121113121111431421132412412112342432431
13126	389.7	124222114313131111124324314312332432111331412
12150	376.4	124431413112113111124143141411411431442141111
11568	136.2	124432411244111131111344131213412414121224212
10566	228.7	124221411442111421111244223113232411111442223
10072	333.3	124231221141411411431221313324312111423113314
9963	104.0	124432311311134112422431441311111131411111311
9869	163.0	124232412231112322134112332121111431222222312
9797	251.8	124144111131423122421142432431141132331222443
9163	108.5	124141143212411111231111134432222111411434311
9024	111.5	224112311111432423122323413411314431122141344
8810	82.3	124433111121111412412242121411322411221131334
7789	44.3	124133122111342431221123142422231412111131112
7147	497.0	124431413131221414342421134111123123421412343
6699	107.5	124431321222131431111421133434341443441341431
6474	78.6	124441322221411412224311141314424221123442413
6432	358.4	124431441221122342423131333411414321141434124
6361	176.3	124132111311131122343422411224131231131111111
5987	143.0	124133312231222432422331342232422411244111411
5914	216.0	124411332224111231312134444111431433421221112
5615	151.5	124111111122334422133421324143442212323442434
5392	177.7	124243311233112311122423143312222221233312341
5187	164.1	124232221123442111411421322331223411431111431
4906	253.6	124432431111311411343244341414323432411421111
4837	69.3	124234222223141112311221434112421421414244411
4690	183.5	124222311112142124111311341124112244411412222
4485	252.5	124431431111222411424443142422213132442111422
4447	295.9	124424133111122321111431334413443232331142412
4438	167.6	124133111441123221441141111122442124134311421
3894	103.9	124131133123111131234112221441222334141123441
3659	48.7	224124234113314132421213314223122433411112311
3364	91.3	124432221112123411411123122432233422111411431
3260	13.2	124141124343313114332244432412413222224432411
3026	101.3	224144111412411123124312123421231243431412412
2825	66.1	124412434322122414411422412443122231112342132
2792	31.9	124431312141122221122421412413431431222124341
2791	100.6	124442322122421311111223441443132233423443231
2722	43.0	124143143224222311133213443342244412122413123
2622	83.0	124431111421221212123132432312331312413411431
2602	84.8	124122123242111133331342411322442123124344411
2503	19.1	124111341412114424314133221312441411121312122
2422	111.7	124232124312111431323133122433443231412233143
2340	101.5	124424131134424314342134223341114411422321312
2323	65.1	224114111342122432321224333441413444123443332
2260	209.6	124113413311233242421141123124131411244412122
2128	86.2	424431311442411111122131423232113441421221113
2015	98.9	124121341142124131443223443142431143111131311
1943	150.9	424131112343412122421244312411433232123131111
1909	25.7	224221222111111413141111311411131432213222231
1853	23.9	224111121441122421442432131441212441443133411
1823	89.1	224142111313112244134324411133314323424313112
1749	75.9	424114222231411411243343142431433431411244244
1691	81.2	124213121222311432412314322142433111413131213
1389	66.0	424111112111244324232222222234442114423314322
1188	56.3	124111131413244244114412431424424213311122213
1086	38.3	124141114213411422124213122123343331113122413
1056	24.0	424411422132413421141221343343424112133412442
879	73.9	124434311124131124122311431123134412434221343
401	43.4	124411111111334221132332141314122421412344221
11168	1	124111111123142113112441331121113131424122323
16221	1	124111111124134421114434432111112133114123431
18189	1	124111111111133311113413421131112134324111324
22251	1	124111114113142212114342434131114131134113433
15253	1	124341114121143341112342433411111414114123432
30310	1	124111111123131342113411311222112134114122311
23499	1	124111111123131131113411311132112134114122311
13662	1	124141114123131131113343311222112411324122311
19557	1	124111111123131131113411311131112134324122311
37183.62	1	124111111123143131113411112222112411324122311
21753.18	1	124111111123134131113122311113112411324122311
26650.18	1	124111111123143131113411311114112411324122444
27677.31	1	124111111123142131113141311114112411324122311
41271.77 	1	124111111123131131113144311113112411324111311
21513.89 	1	124111111123134131113122311414112411324122311
33849.37	1	124111111123234131111124311113112411324122311
38267.66 	1	124111111213143131113141311114112411324122311
8429.355 	1	124111111123143131113121311223112324324122311
36119.64 	1	124111114123131131113144311113112411324122311
25948.7632	1	124111111123423131113411311222112411324111311
22004.69818	1	124111111123424131113144311222112411221111311
36821.22053	1	124111111123131341111411311222411411324111144
29816.23873	1	124111111123423114113411311222112411321111311
10776.21417	1	124111111123131131113411311222112212111111311
26986.89069	1	124111111123423131113411311222112411324111311
14931.57046	1	124111111123423131113411311121112411324111311
15279.68249	1	124111111123423421113144311113112212131111311
32678.01402	1	124111111123414311113144311113112213221111311
35158.60897	1	124111111123423131113144311113112411111111311
10910.95593	1	124111111123131131113411332113311113324334311
43697.7899	1	124111111123131112113144311113311131324111311
16106.68289	1	124111111123131314113144112113311112324111311
49837.88327	1	124111111111131131113144311113311112324111311
15230.64615	1	124111111123131424113144311113311112324111311
17562.67	1	124111111123131314113441311113311113324111144
32959.68247	1	124111111123131131113111423113311132324111311
51555.69857	1	124111111123131113113144311113311131324111311
43073.46206	1	124111111123131131113111311113112411423111314
13776.76503	1	124111111123131131122144311113112411132111141
33663.20277	1	124111111113131113113144311113314131324111311
37441.24898	1	124111111113131113113144311113314131321111311
42801.26973	1	124111111113131113113144311113311131323111311
18987.68459	1	124111111123131342443144311113311131324111311
37894.61442	1	124111111122131113111144311113311131324111311
37426.27819	1	124111111112234113411144311113311131324111311
36062.60361	1	124111111114131113113144311113314131324111311
38667.99479	1	124111111113131113113344311113311131324111311
32477.92257	1	124111111112131113113144311113314131324111323
44898.1892	1	124111111121131113111144311113311131324111311
35738.86602	1	124111111123131113113123311113311131324111311
26284.53333	1	124111111123131113131144311113311131324111224
26564.93538	1	124111111123131113112123311113311131324111311
70490.56471	1	124111111123131113113144311113311131324111442
33831.49307	1	124111111123131113113144311113311131324111224
44175.90095	1	124111111123131113113344311113311131324111311
45539.84505	1	124111111123131113113144311113311131324111311
33569.06909	1	124111111123131113113144311113311131324111322
43290.16667	1	124111111123131113113144311113311131323111311
23680.8	1	124111111123131113113443311113311131324111311
39553.87784	1	124111111123313113113144311113311131324111442
22649.26098	1	124111111123133113113144311113311131324111442
25591.59073	1	124111111213121113113144311113311131324111442
51681.71912	1	124111111213131113113144311113311131324111442
53523.07955	1	124111111123131113113144311113311131134111442
26102.64759	1	124111111123422113113144311113311131324111442
37781.39362	1	124111111213131113411144311111311131324111442


"""

"""
35738.86602 1	124111111123131113113123311113311131324111311
26284.53333 1   124111111123131113131144311113311131324111224
26564.93538 1   124111111123131113112123311113311131324111311
70490.56471 1   124111111123131113113144311113311131324111442
33831.49307 1   124111111123131113113144311113311131324111224
44175.90095 1   124111111123131113113344311113311131324111311
45539.84505 1   124111111123131113113144311113311131324111311
33569.06909 1   124111111123131113113144311113311131324111322
43290.16667 1   124111111123131113113144311113311131323111311
23680.8 1   124111111123131113113443311113311131324111311
"""
data_list = data_str.strip().split('\n')

columns = data_list[0].split('\t')

data = [line.split('\t') for line in data_list[1:]]

df = pd.DataFrame(data, columns=columns)

df['Genèses'] = np.array(df['Genèses'].apply(list))

np_data = df.to_numpy()
#print(np_data)
k=3
all_tokens = []
for label,var, sequence in np_data:
    tokens = [''.join(sequence[i:i+k]) for i in range(len(sequence)-k+1)]
    all_tokens.append(tokens)

#print(all_tokens[0])

model = gensim.models.Word2Vec.load("../test_model_size/origin_embedding")

j=0
embeddings_final = []
for tokens in all_tokens:
    #print(tokens)
    embeddings = np.zeros((len(tokens), 50))
    for i, token in enumerate(tokens):
        if token in model.wv:
            embeddings[i] = model.wv[token]

    pos_enc = positional_encoding(len(tokens), 50).numpy()
    embeddings += pos_enc
    embeddings_final.append(embeddings)
    #break


train_x = []
label_x = []
i=0
t=5
for item in np_data:
    n = 100*t
    if float(item[1])==1:
        n=5000*t
    if float(item[0])>30000:
        n = 5000*t
    if  10000< float(item[0]) < 30000:
        n=500*t

    
    for time in range(n):
        #tr = list(map(float, item[2]))
        tr = embeddings_final[i]
        train_x.append(np.array(tr))
        #la = float(item[0])
        la = float(item[0])+float(item[1])*np.random.uniform(-1, 1)
        label_x.append(la)
    #print(np.random.uniform(-1, 1))
    i+=1
    
train_x = np.array(train_x)
label_x = np.array(label_x)
#print(train_x[0],label_x[0])
print(train_x.shape,label_x.shape)
np.savez("new_encoding_training_1025.npz", train_x = train_x,label_x = label_x)
