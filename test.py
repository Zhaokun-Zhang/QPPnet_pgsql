from parse import * 
from data.tpch_utils import PSQLTPCHDataSet
from model.QPPnets import QPPNet_t, QPPNet_m
from utils import *


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.test_time = True
    opt.type = 'latency'

    if opt.dataset == "PSQLTPCH":
        dataset = PSQLTPCHDataSet(opt)
    print("dataset_size", dataset.datasize)


    if opt.type == 'latency':
        qpp = QPPNet_t(opt)
    else:
        qpp = QPPNet_m(opt)

    qpp.load("best")

    offset = 0
    query = 1
    rq_results = []
    for grpnum in dataset.num_grps:
        print('-'*90)
        print(f'query: {query} evaluation.')
        query += 1
        test_data_q = dataset.test_dataset[offset:offset+grpnum]
        offset += grpnum
        qpp.evaluate(test_data_q)
        rq_results.append(qpp.last_rq)

    print('-'*90,'\n')
    qpp.evaluate(dataset.test_dataset)
    test_rq = qpp.last_rq

    print('query rq eval:')
    for i, rq in enumerate(rq_results):
        print(f"  query: {i+1}, rq: {rq}")
    print('test rq eval: ', test_rq)
