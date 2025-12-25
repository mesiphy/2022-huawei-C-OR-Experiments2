import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
data_1 = pd.read_excel('data/附件1.xlsx')
data_2 = pd.read_excel('data/附件2.xlsx')
data_1 = np.array(data_1)
data_2 = np.array(data_2)
data_1[:,2] = np.where(data_1[:,2]=='燃油',0,1)
data_1[:,3] = np.where(data_1[:,3]=='两驱',0,1)
data_2[:,2] = np.where(data_2[:,2]=='燃油',0,1)
data_2[:,3] = np.where(data_2[:,3]=='两驱',0,1)


class GWO():
    def __init__(self,dim=6000,maxiter=200,size=40,lb=0,ub=0.999999999):
        self.dim = dim
        self.X = np.random.uniform(low=lb,high=ub,size=(size,dim))
        self.alpha_wolf_pos = None
        self.beta_wolf_1_pos = None
        self.beta_wolf_2_pos = None
        self.maxiter = maxiter
        self.size = size
        self.lb = lb
        self.ub = ub
        self.gen_best_y = np.zeros(shape=(self.maxiter+1,))
        self.gen_best_X = np.zeros(shape=(self.maxiter+1,dim))
        self.gen_min_y = None
        self.gen_min_x = None
        self.time_in_out = np.array([18, 12, 6, 0, 12, 18]) / 3
        self.time_fan = np.array([24, 18, 12, 6, 12, 18]) / 3
        self.timeChedaoToHyj = self.time_in_out / 2
        self.time_fanToChedao = np.array([4, 3, 2, 1, 1, 2])
        self.time_0 = 3
        self.gold1 = None
        self.gold2 = None
        self.gold3 = None
        self.gold4 = None

    def fitness(self, data):
        self.curY = np.zeros(shape=(self.size,))

        for i in range(self.size):
            ans, outLine, needNunberFan, costtime = self.genX(maxiter=self.dim / 2, data=data, renwu=self.X[i], cur=i)
            curGold, self.gold1, self.gold2, self.gold3, self.gold4 = self.getGold(outLine, needNunberFan, data, costtime)
            self.curY[i] = -1 * curGold

    def getCurWolf(self, t):
        index = self.curY.argsort()
        self.alpha_wolf_pos = self.X[index[0]]
        self.beta_wolf_1_pos = self.X[index[1]]
        self.beta_wolf_2_pos = self.X[index[2]]

    def getNewX(self, t):
        r_1_a = np.random.uniform(low=0, high=1, size=(self.size, self.dim))
        r_1_b_1 = np.random.uniform(low=0, high=1, size=(self.size, self.dim))
        r_1_b_2 = np.random.uniform(low=0, high=1, size=(self.size, self.dim))
        r_2_a = np.random.uniform(low=0, high=1, size=(self.size, self.dim))
        r_2_b_1 = np.random.uniform(low=0, high=1, size=(self.size, self.dim))
        r_2_b_2 = np.random.uniform(low=0, high=1, size=(self.size, self.dim))
        a = 2 * (1 - t / self.maxiter)
        A_a = 2 * a * r_1_a - a
        A_b_1 = 2 * a * r_1_b_1 - a
        A_b_2 = 2 * a * r_1_b_2 - a
        C_a = 2 * r_2_a
        C_b_1 = 2 * r_2_b_1
        C_b_2 = 2 * r_2_b_2
        distance_a = np.abs(C_a * self.alpha_wolf_pos - self.X)
        distance_b_1 = np.abs(C_b_1 * self.beta_wolf_1_pos - self.X)
        distance_b_2 = np.abs(C_b_2 * self.beta_wolf_2_pos - self.X)
        X_a = self.alpha_wolf_pos - A_a * distance_a
        X_b_1 = self.beta_wolf_1_pos - A_b_1 * distance_b_1
        X_b_2 = self.beta_wolf_2_pos - A_b_2 * distance_b_2
        new_X = (X_a + X_b_1 + X_b_2) / 3
        new_X = np.where(new_X < self.lb, self.lb, new_X)
        new_X = np.where(new_X > self.ub, self.ub, new_X)
        return new_X

    def chedaoOneToOne(self, x, result, time_i):
        xAfter0 = x[:, 3:-1]
        xIndex = np.where(xAfter0 != -1)
        curVar = (xIndex[0].reshape(-1, 1) + 1) * 10 + 10 - (xIndex[1].reshape(-1, 1) + 3) // 3
        curVar = curVar.reshape(-1, 1)
        result[xAfter0[xIndex[0], xIndex[1]].astype(int), time_i * 3:time_i * 3 + 3] = curVar
        xIndex = np.where(x[:, :3] != -1)
        result[x[xIndex[0], xIndex[1]].astype(int), time_i * 3:time_i * 3 + 3] = (xIndex[0].reshape(-1,
                                                                                    1) + 1) * 100 + 10
        xIndex = np.where(x[:, -1] != -1)
        result[x[xIndex[0], -1].astype(int), time_i * 3:time_i * 3 + 3] = (xIndex[0].reshape(-1, 1) + 1) * 10 + 1
        return result

    def fanOneToOne(self, x, result, time_i):  ###x 表示每个车道车位上对应的车的编号， -1 为没车
        curVar = x[0]

        if curVar != -1:
            result[int(curVar), time_i * 3:time_i * 3 + 3] = 710  ###更新反车道 10 车位

        curVar = x[1:]
        curVarIndex = np.where(curVar != -1)[0]  ###找到有车的车位
        result[curVar[curVarIndex].astype(int), time_i * 3:time_i * 3 + 3] = 70 + 9 - curVarIndex.reshape(-1,
                                                                                                          1) // 3  ###更新反车道 1-9 车位
        return result

    def Outputresult(self, x):
        x_Index = np.where(x != 3)[1].max()

        return x[:, :x_Index + 3]

    def getGold(self, x, number, data, costTime):
        l = len(x)

        data = data[:, 2:]
        x_list = []
        for i in range(l):
            x_list.append(data[x[i], 1])
        item2 = self.z2(x_list)
        x_list = []
        for i in range(l):
            x_list.append(data[x[i], 0])
        item1 = self.z1(x_list)
        result = (100 - item1) * 0.4 + (100 - item2) * 0.3 + (100 - number) * 0.2 + 0.1 * (100 - (costTime - 2934) * 0.01)
        return result, (100 - item1) * 0.4, (100 - item2) * 0.3, (100 - number) * 0.2, 0.1 * (
                    100 - (costTime - 2934) * 0.01)

    def z1(self, data):
        result = 0

        cc = 0
        data = np.array(data)
        a = np.where((data == 1))
        for i in range(1, len(a[0])):
            if a[0][i] - a[0][i - 1] == 3:
                result = result + 1
            else:
                cc = cc + 1
        rrr = cc

        return rrr

    def z2(self, list1):
        result = [0]

        rr = []
        flag = 0
        j = 1
        rrr = 0
        R = 0
        for i in range(1, len(list1)):
            if list1[i] == list1[i - 1]:
                flag += 1
            else:
                flag = 0
            result.append(flag)
        result.append(0)
        for i in range(1, len(result)):
            if result[i] <= result[i - 1]:
                rr.append(result[i - 1] + 1)
        while j < len(rr):
            if rr[j] == rr[j - 1]:
                rrr = rrr + 1
            else:
                R = R + 1
            j = j + 2
        if len(rr) % 2:
            R = R + 1
        return R

    def genX(self, renwu, maxiter, data, cur):
        renwu = renwu.reshape(-1, 2)

        m, n = data.shape  ###数据长度
        outputLine = []  ###输出队列
        inputHyjToChedaoGang = []  ###刚刚从输入横移机放入车道
        outputHyjTofanGang = []  ###刚刚从输出横移机放入反车道
        result = np.zeros(shape=(m, int(maxiter * 3)))
        chedaoIndex = np.zeros(shape=(6, 28)) - 1  ###车道上车的编号，没车就是-1
        fanIndex = np.zeros(28) - 1  ###返回道上车位的编号，没车为-1

        chedaoState = np.zeros(shape=(6, 28))  ###车道上有无车，有车 1，没车 0
        fanState = np.zeros(28)  ###反车道上的状态，有车 1，没车 0
        fanNunber = 0  ###使用返回道次数
        resultHyjStarEnd = np.zeros(shape=(m, 5))  ###0 还没出， 1 在横移机上， 2 在车道上， 3 在输出横移机， 4 在终

        resultHyjStarEnd[:, 0] = 1
        ifStar = np.zeros(m)
        ifStar[0] = 1  ###
        if_fanStar = 0  ###返回车道的第 10 车位状态， 0 为空， 1 为满
        ifInputHyj = 0  ###入口横移机是否在工作， 0 为空， 1 为满
        ifOutputHyj = 0  ###出口横移机是否在工作， 0 为空， 1 为满
        timeFanArrive = 0  ###返回道到达时间
        fromWhere = 0  ###输入横移机上的车的来源， 0 表示 PBS， 1 表示返回道
        arriveIndex = []  ###到达最后一个车位的车道顺序
        outOrInput = 0  ###横移机准备把车送出去还是送回返回道， 0 表示去返回道， 1 表示送出去
        inputHyjCar = -1  ###初始化输入横移机上的车的编号， -1 表示没有车
        OutHyjCar = -1  ###初始化输出横移机上的车编号，没车表示-1
        PbsCarCur = 0  ###初始化 PBS->输入横移机上的编号，第一辆为 0
        arrive10Chewei = np.zeros(m)
        i = 0
        while i < maxiter:
            ifRunChedao = np.zeros(shape=(6, 27))  ###可移动的车道,1 表示可移动， 0 表示不可移动,这得对约束的理解
            ifRunFan = np.zeros(27)  ###返回道是否可移动,1 表示可移动， 0 表示不可移动
            if ifInputHyj == 0:  ###如果输入横移机空闲
                ifRunChedao_0 = chedaoState[:, :3]  ###10 车位上的车道状态
                ifRunChedao_0 = np.where(ifRunChedao_0 == 1)  ###10 车位上有车的索引
                needQueren = []
                canRunChedao = [3, 2, 4, 1, 5, 0]  ###可选择车道
                for j in range(6):
                    if j in ifRunChedao_0[0]:
                        canRunChedao.remove(j)  ###剩下一定能走的车道
                        needQueren.append(j)  ###需要计算时间确认的车道
        ###这里根据来源不同进行不同的计算可选择车道
        ###计算可选择车道这里有问题
        # if if_fanStar == 1: ###10 号车位返回道有车
        # if needQueren:
        # for j in range(len(needQueren)):
        # if time_fanToChedao[needQueren[j]] > 3 - ifRunChedao_0[needQueren[j], 1]:
        # canRunChedao.append(needQueren[j])
                if PbsCarCur < m:
                    if if_fanStar == 0:  ###当返回道的 10 车位为空时，
                    # canRunChedao = [3,2,4,1,5,0] ###输入横移机可选择的部分
                        canRunChedao.append('等待')
                        chooseChedao = renwu[i, 0] * len(canRunChedao)  ###根据既定任务选择车道
                        chooseChedao = canRunChedao[int(chooseChedao)]  ###选择进哪个车道
                if if_fanStar == 1:  ###返回车道不为空
                    chooseChedao = renwu[i, 0] * len(canRunChedao)  ###根据既定任务选择车道
                    chooseChedao = canRunChedao[int(chooseChedao)]

                if chooseChedao != '等待':  ###如果不选择等待
                    timeInputHyjNeed = self.time_in_out[chooseChedao]  ###所选择的车道需要花的时间
                    timeInputHyjWorkCur = i  ###输入横移机开始工作的时间
                    if PbsCarCur < m:  ###仓库内还有车
                        if if_fanStar == 0:  ###如果返回道 10 车位没车
                            ifInputHyj = 1  ###立刻可以拿车，将横移机置为工作状态
                            fromWhere = 0  ###记录数据来源，从仓库来的数据
                            result[PbsCarCur, i * 3:(i + 1) * 3] = 1  ###记录数据
                            inputHyjCar = PbsCarCur  ###输入横移机从仓库拿车，更新输入横移机上车的序号
                            PbsCarCur += 1  ###下一辆仓库的出车序号
                if if_fanStar == 1:
                    fromWhere = 1  ######记录数据来源，从反车道来的数据
                    ifInputHyj = 0  ###此时还没装车，为了方便处理所以将输入横移机的工作状态置为 0，实际此时为 1
                    if i >= timeFanArrive + 1:  ###输入横移机到达反车道 10 车位
                        inputHyjCar = fanIndex[0]  ###输入横移机此时携带的车的序号
                        result[int(inputHyjCar), i * 3:(i + 1) * 3] = 1  ###记录结果
                        fanIndex[0] = -1  ###车被搬走，所以返回道 10 车位没有车，用-1 表示
                        fanState[0] = 0  ###车被搬走，返回道 10 车位没车， 0 表示没车
                        if_fanStar = 0  ###车被搬走，返回道 10 车位没车， 0 表示没车
                        timeInputHyjWorkCur = i  ###输入横移机从返回道接受车的时间点
                        ifInputHyj = 1  ###
            if ifInputHyj == 1:  ###如果输入横移机在工作
                if fromWhere == 0:  ###如果从仓库来的
                    if inputHyjCar != -1:
                        result[int(inputHyjCar), i * 3:(i + 1) * 3] = 1  ###记录结果
                    if i == (timeInputHyjWorkCur + timeInputHyjNeed / 2):  ###如果输入横移机送到目标车道
                        inputHyjToChedaoGang.append(chooseChedao)  ###输入横移机刚刚把车放在指定车道
                        chedaoState[chooseChedao, 0] = 1  ###将所选择车道的 10 车位置为 1
                        chedaoIndex[chooseChedao, 0] = inputHyjCar  ###更新车道上的车的序号
                        result[inputHyjCar, i * 3:(i + 1) * 3] = (1 + chooseChedao) * 100 + 10  ###记录结果
                        inputHyjCar = -1  ###输入横移机此时不再有车

                    if i >= timeInputHyjNeed + timeInputHyjWorkCur:  ###输入横移机回到原来位置时
                        ifInputHyj = 0  ###输入横移机工作转态置为 0
                        timeFanArrive = i  ###修改的地方#####################
                if fromWhere == 1:
                    if i == timeInputHyjWorkCur + self.time_fanToChedao[int(chooseChedao)]:
                        chedaoState[chooseChedao, 0] = 1  ###从返回道的 10 车位来的车，由输入横移机放到选择的车道中
                        chedaoIndex[chooseChedao, 0] = inputHyjCar  ###更新被选择的车道上车的编号
                        inputHyjToChedaoGang.append(chooseChedao)  ###输入横移机刚刚把车放在指定车道
                        result[int(inputHyjCar), i * 3:i * 3 + 3] = chooseChedao * 100 + 10  ###更新结果
                        inputHyjCar = -1  ###输入横移机上没有车
                    if i >= self.time_fan[chooseChedao] - 1 + timeInputHyjWorkCur:  ###输入横移车回到开始的位置时
                        ifInputHyj = 0
                        timeFanArrive = i  ###修改的地方######################
            if ifOutputHyj == 0 and len(arriveIndex) != 0:  ###如果输出横移机空闲，且 1 车位内有车
                timeOutputHyjWorkCur = i  ###输出横移机开始工作的时间
                timeOutputHyjNeed = self.time_in_out[arriveIndex[0]] / 2  ###输出横移机需要到达指定车道所花的时间
                timeOutputHyjChedaoToFan = self.time_fanToChedao[arriveIndex[0]]  ###输出横移机从指定车道送到返回道的时间
                ifOutputHyj = 1  ###输出横移机工作状态置为 1
            if ifOutputHyj == 1:
                if i == timeOutputHyjWorkCur + timeOutputHyjNeed:  ###输出横移机到达 1 车位
                    ###到达 1 车位后对输出还是回返回道进行判断
                    if fanState[25:].any() == 1:
                        outOrInput = 1
                    if fanState[25:].any() != 1:
                         if renwu[i, 1] <= 0.5:
                            outOrInput = 1
                         if renwu[i, 1] > 0.5:
                            outOrInput = 0

                    result[int(chedaoIndex[int(arriveIndex[0]), -1]), i * 3:i * 3 + 3] = 2  ###车到了输出横移机身上，记录结果
                    OutHyjCar = chedaoIndex[int(arriveIndex[0]), -1]  ###输出横移机上面车的序号
                    chedaoState[int(arriveIndex[0]), -1] = 0  ###车道状态改变
                    chedaoIndex[int(arriveIndex[0]), -1] = -1  ###车道上对应车位的车的序号置为-1（-1 表示没有车）
                    arriveIndex.pop(0)  ###到达最后一个车位的车道顺序里去除被接走的车


                if i >= timeOutputHyjWorkCur + timeOutputHyjNeed:
                    ###outOrInput 这个规则及约束还没写
                    if outOrInput == 0:  ###把车送到返回道
                        if i < timeOutputHyjWorkCur + timeOutputHyjNeed + timeOutputHyjChedaoToFan:  ###车在输出横移机上
                            result[int(OutHyjCar), i * 3:i * 3 + 3] = 2
                        if i == timeOutputHyjWorkCur + timeOutputHyjNeed + timeOutputHyjChedaoToFan:  ###把车从输出横移机放到返回道
                            arrive10Chewei[int(OutHyjCar)] += 1  ###车到达返回道，这表明车已用掉一次返回的机会
                            result[int(OutHyjCar), i * 3:i * 3 + 3] = 71
                            outputHyjTofanGang.append(27)  ###刚刚把车从输出横移机放到反车道
                            fanState[-1] = 1  ###返回车道的 1 车位放车
                            fanIndex[-1] = OutHyjCar  ###返回车道上的 1 车位上车的序号
                            OutHyjCar = -1  ###输出横移机不再有车
                            fanNunber += 1
                        if i == timeOutputHyjWorkCur + timeOutputHyjNeed + timeOutputHyjChedaoToFan + 1:  ###输出横移机结束工作
                            ifOutputHyj = 0
                    if outOrInput == 1:
                        if i < timeOutputHyjWorkCur + 2 * timeOutputHyjNeed:  ###输出横移机还未到达指定车道，车还在横移机上
                            result[int(OutHyjCar), i * 3:i * 3 + 3] = 2
                        if i == timeOutputHyjWorkCur + 2 * timeOutputHyjNeed:  ###输出横移机成功把车送出去
                            result[int(OutHyjCar), i * 3:i * 3 + 3] = 3  ###更新记录
                            outputLine.append(int(OutHyjCar))  ###输出队列添加上刚刚送出去的车
                            ifOutputHyj = 0  ###输出横移机结束工作
                            OutHyjCar = -1  ###输出横移机上没有车，置为-

                    ###车道能否移动判断矩阵
            for j in range(6):
                for j_index in range(9):
                    if j_index == 0:
                        if chedaoState[j, -1] == 0:  ###1 车位没车的话，这条道一定可以走
                            ifRunChedao[j] = 1
                            break
                    if j_index > 0:
                        if chedaoState[j, 27 - j_index * 3:30 - j_index * 3].any() == 0:  ###当前车位没车的话，后面的车一定可以往前走
                            ifRunChedao[j, :27 - j_index * 3] = 1
                            break

            ifRunChedao = np.append(ifRunChedao, np.zeros(shape=(6, 1)), axis=1)  ###1 车位不能通过车道往前走
            ###返回道可否移动的判断矩阵
            for j in range(9):
                if j == 0:
                    if fanState[0] == 0:  ###返回道 10 车位没车时，返回道都可移动
                        ifRunFan[:] = 1
                        break
                if j > 0:
                    if fanState[j * 3 - 2: j * 3 + 1].any() == 0:  ###返回道当前车位没车时，小于该车位的车位可移动
                        ifRunFan[j * 3:] = 1
                        break

            ifRunFan = np.append(np.zeros(1), ifRunFan)  ###返回道 10 车位的车不可通过返回道移动
            haveCarCanGo = ifRunChedao * chedaoState  ###有车且能走的车
            if inputHyjToChedaoGang:
                haveCarCanGo[int(chooseChedao), 0] = 0  ###刚刚到车道的车不能走
                inputHyjToChedaoGang.pop()  ###取消刚刚到车道这个‘刚刚’的状态
            haveCarCanGoIndexOldChedao = np.where(haveCarCanGo == 1)[0]
            haveCarCanGoIndexOldChewei = np.where(haveCarCanGo == 1)[1]
            haveCarCanGoIndexNewChewei = haveCarCanGoIndexOldChewei + 1
            chedaoState[haveCarCanGoIndexOldChedao, haveCarCanGoIndexNewChewei] = 1
            chedaoState[haveCarCanGoIndexOldChedao, haveCarCanGoIndexOldChewei] = 0

            chedaoIndex[haveCarCanGoIndexOldChedao, haveCarCanGoIndexNewChewei], chedaoIndex[
                haveCarCanGoIndexOldChedao, haveCarCanGoIndexOldChewei] = \
                chedaoIndex[haveCarCanGoIndexOldChedao, haveCarCanGoIndexOldChewei], chedaoIndex[
                    haveCarCanGoIndexOldChedao, haveCarCanGoIndexNewChewei]
            result = self.chedaoOneToOne(chedaoIndex, result, i)  ###记录车道上车的位置
            chedaoLastChewei = chedaoState[:, -1]  ###1 车位上的状态
            chedaoLastChewei = np.where(chedaoLastChewei == 1)[0]  ###1 车位上有车的车道

            for j in chedaoLastChewei:
                if j not in arriveIndex:  ###已经在到达 1 车位的队列则不再添加
                    arriveIndex.append(j)  ###到达 1 车位的队列添加新到达的

            ###反车道状态改变
            haveFanCarCanGo = ifRunFan * fanState  ###反车道上有车且能走的车
            if outputHyjTofanGang:
                haveFanCarCanGo[-1] = 0  ###刚刚到反车道的车不能动
                outputHyjTofanGang.pop()  ###取消刚刚到反车道的这个‘刚刚的’状态
            haveCarCanGoIndexOldfan = np.where(haveFanCarCanGo == 1)[0]  ###反车道上有车的车位
            haveCarCanGoIndexNewfan = haveCarCanGoIndexOldfan - 1  ###反车道上的车移动后的车位
            fanState[haveCarCanGoIndexOldfan] = 0
            fanState[haveCarCanGoIndexNewfan] = 1

            if fanState[0] == 1:  ###返回道 10 车位有车时
                if_fanStar = 1  ###返回道 10 车位修改为有车

            fanIndex[haveCarCanGoIndexOldfan], fanIndex[haveCarCanGoIndexNewfan] = \
                fanIndex[haveCarCanGoIndexNewfan], fanIndex[haveCarCanGoIndexOldfan]  ###调整反车道上车的序号
            if fanState[0] == 1:
                if_fanStar = 1
            if fanState[0] == 0:
                if_fanStar = 0

            if inputHyjCar != -1:
                result[int(inputHyjCar), i * 3:i * 3 + 3] = 1
            if OutHyjCar != -1:
                result[int(OutHyjCar), i * 3:i * 3 + 3] = 2

            result = self.fanOneToOne(fanIndex, result, i)  ###记录返回道上车的位置
            result[outputLine, i * 3:i * 3 + 3] = 3  ###记录车出去后，车所在的位置

            if i > 750:
                if len(np.where(result[:, i * 3] != 3)[0]) == 0:
                    result = result[:, :(i + 1) * 3]
                    break

            if i == maxiter - 2:
                # # print(0)
                # # print(i)
                # self.X[cur] = np.random.uniform(low=self.lb, high=self.ub, size=(self.dim,))
                # renwu = self.X[cur]
                # renwu = renwu.reshape(-1, 2)
                # # i = 0
                break
                #
                # outputLine = []  ###输出队列
                # inputHyjToChedaoGang = []  ###刚刚从输入横移机放入车道
                # outputHyjTofanGang = []  ###刚刚从输出横移机放入反车道
                # result = np.zeros(shape=(m, int(maxiter * 3)))
                # chedaoIndex = np.zeros(shape=(6, 28)) - 1  ###车道上车的编号，没车就是-1
                # fanIndex = np.zeros(28) - 1  ###返回道上车位的编号，没车为-1
                # chedaoState = np.zeros(shape=(6, 28))  ###车道上有无车，有车 1，没车 0
                # fanState = np.zeros(28)  ###反车道上的状态，有车 1，没车 0
                # fanNunber = 0  ###使用返回道次数
                # resultHyjStarEnd = np.zeros(shape=(m, 5))  ###0 还没出， 1 在横移机上， 2 在车道上， 3 在输出横移机， 4在终点
                # resultHyjStarEnd[:, 0] = 1
                # ifStar = np.zeros(m)
                # ifStar[0] = 1  ###
                # if_fanStar = 0  ###返回车道的第 10 车位状态， 0 为空， 1 为满
                # ifInputHyj = 0  ###入口横移机是否在工作， 0 为空， 1 为满
                # ifOutputHyj = 0  ###出口横移机是否在工作， 0 为空， 1 为满
                # timeFanArrive = 0  ###返回道到达时间
                # fromWhere = 0  ###输入横移机上的车的来源， 0 表示 PBS， 1 表示返回道
                # arriveIndex = []  ###到达最后一个车位的车道顺序
                # outOrInput = 0  ###横移机准备把车送出去还是送回返回道， 0 表示去返回道， 1 表示送出去
                # inputHyjCar = -1  ###初始化输入横移机上的车的编号， -1 表示没有车
                # OutHyjCar = -1  ###初始化输出横移机上的车编号，没车表示-1
                # PbsCarCur = 0  ###初始化 PBS->输入横移机上的编号，第一辆为 0
                # arrive10Chewei = np.zeros(m)
            i += 1
        return result, outputLine, fanNunber, result.shape[1]

    def run(self, data):
        t1 = time.time()

        for i in range(self.maxiter):
            self.fitness(data)
            self.gen_best_y[i] = np.min(self.curY)
            self.gen_best_X[i, :] = self.alpha_wolf_pos
            self.getCurWolf(i)
            self.X = self.getNewX(i)
        # if i >10 and self.gen_best_y[i] > self.gen_best_y[i-10]-0.0001:
        # break
            if i % 10 == 0:
                print(i)
            t2 = time.time()
            if t2 - t1 > 6000:
                break

        self.fitness(data)
        self.gen_best_y[i + 1] = np.min(self.curY)
        self.gen_best_y = self.gen_best_y[:i + 1]
        self.gen_best_X[i + 1, :] = self.alpha_wolf_pos
        self.gen_best_X = self.gen_best_X[:i + 1, :]
        index = np.argmin(self.curY)
        best_x = self.X[index]
        best_y = self.curY[index]
        gen_min_y_index = self.gen_best_y.argmin()
        self.gen_min_y = self.gen_best_y[gen_min_y_index]
        self.gen_min_x = self.gen_best_X[gen_min_y_index]
        return best_x, best_y


# model = GWO(maxiter=400)
# best_x, best_y = model.run(data_1)
# plt.plot(model.gen_best_y)
# plt.show()
# result, outputLine, fanNunber, costTime = model.genX(maxiter=model.dim / 2, data=data_1, cur=22,
#                                                      renwu=model.gen_min_x)
# result = pd.DataFrame(result)
# result.to_excel('附件 1 的结果.xlsx')

model = GWO(maxiter=400)
best_x, best_y = model.run(data_2)
plt.plot(model.gen_best_y)
plt.show()

result, outputLine, fanNunber,costTime=model.genX(maxiter=model.dim/2,data=data_1,cur=22,renwu=model.gen_min_x)
result = pd.DataFrame(result)
result.to_excel('附件 2 的结果.xlsx')
