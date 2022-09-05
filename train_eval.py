# coding: UTF-8
import numpy as np
import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
import datasets
import transforms
from torch.utils.data import DataLoader
import datetime
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score,accuracy_score,classification_report,confusion_matrix
import pickle as pkl
from torch.nn.utils import clip_grad_norm_
from clearml import Task, OutputModel
task = Task.get_task(project_name='HPO Auto-training', task_name='Datatype Classifier')
logger = task.get_logger()

def train(config, model, train_iter, val_iter, test_iter):
    with open('./DC/data/index2labels.pkl', 'rb') as f:
        index2labels = pkl.load(f)
    labels = {}
    for k, v in index2labels.items():
        labels[v] = k
    
    if config.load_model_path:
        model.load(config.load_model_path)

    print("使用设备：",config.device)
    #if torch.cuda.device_count() > 1: #使用多GPU进行训练
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = torch.nn.DataParallel(model)

    model.to(config.device)
    model.train() #训练模式
    trial_id = nni.get_sequence_id()

    # step3: 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                             lr = config.lr,
                             weight_decay = config.weight_decay)

    # 学习率衰减方案1：每epoch第n_epoch次，按lr_decay衰减
    # scheduler = lr_scheduler.StepLR(optimizer,config.n_epoch,config.lr_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch,eta_min=4e-08)
    # 学习率衰减方案2：自适应衰减
    # 如果一轮epoch训练结束 相较上轮验证集上的f1-score没有上升， 学习率减半。也可以把忍耐值patience设置的大一些 学习率最小减到min_lr为止
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=1, min_lr=5e-6, verbose=True)
    
    # 训练
    batch_count = 0
    best_acc_val = 0.0
    flag = False
    last_improve = 0
    best_epoch = 0
    for epoch in range(config.max_epoch):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        if config.use_lrdecay and epoch > 0:
            scheduler.step()
        if batch_count >= config.max_iter:
            print('max iteration reached....so training is over...')
            break
        iter_time_sum = 0
        iter_count_per = 0
        for X,seq_lengths,y in train_iter:
            iter_start = time.time()
            batch_count += 1
            iter_count_per += 1 
            
            X = X.to(config.device)
            seq_lengths = seq_lengths.to(config.device)
            y = y.to(config.device)

            y_hat = model(X)

            loss = criterion(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            if config.use_clip: #梯度裁剪
                clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            # if config.use_lrdecay:
            #     scheduler.step()
            train_l_sum += loss.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            iter_end = time.time()
            iter_time = iter_end - iter_start
            iter_time_sum = iter_time_sum + iter_time
            if batch_count % 1000 == 1:
                # iter_test_f1,iter_test_acc = evaluate_accuracy(test_iter, model)
                # print('epoch %d, iter %d, iter_test_f1 %.3f, iter_test_acc %.3f, iter_time %.3f' % (epoch + 1, iter_count, iter_test_f1, iter_test_acc, iter_end - iter_start))
            #else:
                print('epoch %d, iter %d, iter_time %.3f' % (epoch + 1, batch_count, iter_end - iter_start))
                logger.report_scalar(title=f'Loss', 
                                     series=f'trial_{trial_id}',
                                     value=loss.item(),
                                     iteration=batch_count)
            if batch_count >= config.max_iter:
                #print('max iteration reached....so training is over...')
                break

            if batch_count - last_improve > config.require_improvement: #如果长期没有提高 就提前终止
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

        # #一个epoch后在验证集上做一次验证
        # print('VAL evaluation.....')
        # val_f1,val_acc = evaluate_accuracy(val_iter, model)

        train_acc_1 = 0
        train_loss_1 = 0
        #train_acc_1, train_loss_1 = evaluate(model, train_iter, test=False) #在整个训练集上的结果，如无必要不需要evaluate
        val_acc, confusion = test(config, model, val_iter)
        
        # 超参搜索 NNI HPO by hjy
        nni.report_intermediate_result(val_acc)
        logger.report_scalar(title=f'Validation accuracy', series=f'trial_{trial_id}', value=val_acc, iteration=epoch)
        # 超参搜索 NNI HPO by hjy
        
        print("Saving on one GPU!")#单GPU训练时保存参数
        print(time.asctime())
        #torch.save(model.state_dict(),'./DC/saved_dict/'+ config.model_name + '_' + str(epoch + 1) +'.pth')

        # 如果一轮epoch训练结束 相较上轮验证集上的acc没有上升
        # scheduler.step(val_f1)
        if val_acc > best_acc_val:
            last_improve = batch_count
            best_acc_val = val_acc
            best_epoch = epoch + 1
            # 保存在验证集上weighted average f1最高的参数（最好的参数）
            #if torch.cuda.device_count() > 1: #多GPU训练时保存参数
            #    print("Saving on ", torch.cuda.device_count(), "GPUs!")
            #    model.save_multiGPU()
            #else:
            #print("Saving on one GPU!")#单GPU训练时保存参数
            #print(time.asctime())
            #使用当前最好的参数，在测试集上再跑一遍
            # best_f1_test,best_acc_test = evaluate_accuracy(test_iter,model)
            # best_acc_test = test(model, test_iter)
            print('current model test acc is best', best_acc_val)
            #torch.save(model.state_dict(),'./DC/saved_dict/'+ config.model_name + '_best.pth') #也可以保存总体的acc最高的这个模型，此训练流程中每一轮都保存
            
            # 超参搜索 NNI HPO by hjy
            trial_id = nni.get_sequence_id()
            #torch.save(model.state_dict(), f'./nni/checkpoints/{config.model_name}_trial_{trial_id}.pth')
            torch.jit.script(model).save(f'./nni/checkpoints/{config.model_name}_trial_{trial_id}.pt')
            # 超参搜索 NNI HPO by hjy

        # print('epoch %d, lr %.6f,loss %.4f, train acc %.3f, val acc %.3f,val f1 %.3f, val best acc %.3f,test best acc %.3f,test best f1 %.3f,time %.1f sec'
        #       % (epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], train_l_sum / batch_count, train_acc_sum / n, val_acc,val_f1, best_acc_val,best_acc_test,best_f1_test,time.time() - start))
        if flag:
            break
        print('epoch %d, lr %.6f,loss %.4f,loss1 %.4f, train acc %.3f,acc1 %.3f, val acc %.3f, val best acc %.3f, best epoch %d, time %.1f sec'
              % (epoch + 1, optimizer.state_dict()['param_groups'][0]['lr'], train_l_sum / batch_count,train_loss_1, train_acc_sum / n,train_acc_1, val_acc, best_acc_val, best_epoch, time.time() - start))
        print('time usage of each iteration in this epoch %.5f sec' % (iter_time_sum/iter_count_per))

    # 超参搜索 NNI HPO by hjy
    nni.report_final_result(val_acc)
    logger.report_matrix("Confusion Matrix",
                         "ignored",
                         iteration=int(trial_id),
                         matrix=confusion,
                         yaxis_reversed=True)
    
    output_model = OutputModel(task=task,
                               name=f"Datatype Classifier trial_{trial_id}",
                               framework="PyTorch",
                               config_text=f"Validation accuracy: {val_acc}")
    output_model.update_labels(labels)
    output_model.update_weights(weights_filename=f'./nni/checkpoints/{config.model_name}_trial_{trial_id}.pt', iteration=trial_id)
    # 超参搜索 NNI HPO by hjy


def test(config, model, test_iter):
    # test
    # model.load_state_dict(torch.load(save_path)) #加载使验证集损失最小的参数
    # model.eval() #测试模式
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True) #计算测试集准确率，每个batch的平均损失 分类报告、混淆矩阵
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    #print("Time usage:", time_dif)
    return test_acc, test_confusion

def evaluate(config, model, data_iter, test=False):
    model.eval() #测试模式
    loss_total = 0
    predict_all = np.array([], dtype=int) #存储验证集所有batch的预测结果
    labels_all = np.array([], dtype=int) #存储验证集所有batch的真实标签
    with torch.no_grad():

        for i,(X, seq, y) in enumerate(data_iter):

            X = X.to(config.device)
            y = y.to(config.device)
            seq= seq.to(config.device)
            out = model(X)
            loss = F.cross_entropy(out, y)
            loss_total += loss
            labels = y.data.cpu().numpy()
            predic = torch.max(out.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all) #计算验证集准确率
    if test: #如果是测试集的话 计算一下分类报告和混淆矩阵
         class_list = list(pkl.load(open('./DC/data/index2labels.pkl','rb')).values())
         print(class_list)
         report = metrics.classification_report(labels_all, predict_all, target_names=class_list, digits=4)
         confusion = metrics.confusion_matrix(labels_all, predict_all) #计算混淆矩阵
         model.train()
         return acc, loss_total / len(data_iter), report, confusion
    model.train()
    return acc, loss_total / len(data_iter) #返回准确率和每个batch的平均损失

def evaluate_accuracy(config, data_iter, net):
    #计算模型在验证集上的相关指标 多分类我们使用 weighed average f1-score
    # start = time.time()
    acc_sum, n = 0.0, 0
    net.eval()  # 评估模式, 这会关闭dropout
    y_pred_total = []
    y_total = []
    
    all_time = 0
    for X,seq_lengths,y in data_iter:
        # eval_iter_start = time.time()
        X = X.to(config.device)
        y = y.to(config.device)
        seq_lengths = seg_lengths.to(config.device)
        with torch.no_grad():
            y_hat = net(X,seq_lengths)       
            y_pred = y_hat.argmax(dim=1).cpu().numpy()
            y_pred_total.append(y_pred)
            y_total.append(y.cpu().numpy())
        # all_time += time.time()-eval_iter_start

    # print('all_time',all_time)    
    # print('evaluate_accuracy time usage:', time.time() - start)

    y_pred = np.concatenate(y_pred_total)
    y_label = np.concatenate(y_total)
    weighted_f1 = f1_score(y_label,y_pred,average='weighted')
    accuracy = accuracy_score(y_label,y_pred)
    net.train()  # 改回训练模式
    return weighted_f1,accuracy
