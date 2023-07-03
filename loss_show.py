import numpy as np
import matplotlib.pyplot as plt

log_path = './memory-train_loss.txt'

if __name__ == '__main__':
    with open(log_path, 'r') as f:
        lines = f.readlines()

    train_losses = []
    test_losses = []
    rq_list = []
    for line in lines:
        if line[0]!='e': continue
        items = line.split(';')
        train_loss, test_loss = items[2:4]
        rq = items[6].partition('l')[0]
        train_losses.append(float(train_loss.partition(':')[2]))
        test_losses.append(float(test_loss.partition(':')[2]))
        rq_list.append(float(rq.partition(':')[2]))

    
    train_arr = np.array(train_losses)
    test_arr = np.array(test_losses)
    rq_arr = np.array(rq_list)
    # train_arr = train_arr - train_arr.mean()
    # test_arr = test_arr - test_arr.mean()

    plt.plot(range(len(train_losses)), train_arr, label='train_loss')
    plt.plot(range(len(test_losses)), test_arr, label='test_loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.clf()
    plt.plot(np.arange(rq_arr.shape[0])[-50:], rq_arr[-50:], label='R(q)')
    plt.legend()
    plt.savefig('R(q).png')
    
    for i in np.arange((rq_arr.shape[0]))[-10:]:
        print(f"epoch: {i+1}, r_q: {rq_arr[i]}")


