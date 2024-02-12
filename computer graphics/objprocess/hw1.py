import math



def edgelen(list1, list2):#длина ребра (разность между координатами, корень скалярного произведения)
    sum=0
    for i in range(0, len(list1)):
        sum+=(list2[i]-list1[i])**2
    return math.sqrt(sum)

def contains(lists, list):#функция проверяющая содержится ли ребро в списке
    if list in lists or list[::-1] in lists:
            return True
    return False


def edgeslen(vertexes, faces):#подсчет длины ребер
    usededges=[]#список ребер, длина которых уже подсчитана
    sum=0

    for i in range(0, len(faces)):

        for j in range(0, len(faces[i])):
            for k in range(j+1, len(faces[i])):
                edge=[faces[i][j], faces[i][k]]#ребро данной грани
                if(not contains(usededges, edge)):#если длина ребра еще не была подсчитана, считаем длину
                    sum+=edgelen(vertexes[edge[0]-1], vertexes[edge[1]-1])#сумма длин
                    usededges.append(edge)#добавляем подсчитанное ребро
    return sum

def vertexmax(vertexes, faces):#вершина, которая принадлежит максимальному количеству граней
    vertexnum=[]#номера вершины
    numoftimes=0#количество вершин

    for i in range(0, len(vertexes)):#проходим через вершины
        numoftimes1 = 0
        for face in faces:
            for j in range(0, len(face)):#проверяем принадлежит ли вершина данной грани
                if face[j]==i+1:
                    numoftimes1=numoftimes1+1#если принадлежит увеличивем количество граней содержащих вершину
                    break
        if numoftimes<=numoftimes1:#выбираем максимальное количество
            if numoftimes==numoftimes1:
                vertexnum.append(i)
            else:
                numoftimes=numoftimes1
                vertexnum=[]
                vertexnum.append(i)
    return (vertexnum,numoftimes)#возвращаем номер вершины и количество граней которые ее содержат

with open("teapot.obj", "r") as f:
    data=f.readlines()

    vertexes=[]
    faces=[]
    for i in range(0, len(data)):
        el=data[i]
        if el[0]=="v":
            vertexes.append(list(map(float, el.replace("\n", "").split(" ")[1:])))
        else:
            faces.append(list(map(int, el.replace("\n", "").split(" ")[1:])))

    print(edgeslen(vertexes, faces))#информация о длине ребер
    info=vertexmax(vertexes, faces)
    for vert in info[0]:
        print(vertexes[vert], info[1])#информация вершине с наибольшим количеством граней