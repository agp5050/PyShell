#coding=utf-8
import os,sys
import ConfigParser
from pyhive import hive
import pandas as pd
import time
import zipfile
import os
from Crypto.Cipher import PKCS1_v1_5 as Cipher_pkcs1_v1_5
from Crypto.PublicKey import RSA
import paramiko
# retrun dataframe
class SftpZipedFile:
    def __init__(self):
        self.txtFileList=[]
        self.typeSign = {'债券融资': ('12','121EXPORTTRADEINFO.txt'), '删除': ('00','000DELETEINFO.txt')}
        self.public_key_file_name = os.path.dirname(args[0]) + '/Key_pub.pem'
        self.private_key_file_name = os.path.dirname(args[0]) + '/Key.pem'
    def readFromHive(self,sql):
        confPsr = ConfigParser.ConfigParser()
        global args
        ConfPath = os.path.split(args[0])[0] + '/hiveConf.ini';
        confPsr.read(ConfPath)
        hostName = confPsr.get('base_info', 'HOST')
        portN = confPsr.get('base_info', 'PORT')
        Uname = confPsr.get('base_info', 'USERNAME')
        passW = confPsr.get('base_info', 'PASSWORD')
        DBName = confPsr.get('base_info', 'DATABASE')
        try:
            conn = hive.Connection(host=str(hostName), port=int(portN), username=str(Uname), password=str(passW),
                                   auth='LDAP',
                                   thrift_transport=None, database=DBName)
            # cusr=conn.cursor()
            # cusr.execute()
            # cusr.fetchall() 返回list  然后open file . file.write('%s \r \n'％list)也可以。
            df = pd.read_sql(sql, conn)
            return df
        except Exception, e:
            print 'Hive Error %d : %s' % (e.args[0], e.args[1])

    def df2file(self,df, fileType='债券融资'):
        print df.shape
        def encod(x):
            if type(x) == str:

                return x.decode('utf-8', errors='ignore').encode('gb18030', errors='ignore')
            else:
                return x
        print df.iloc[:,8]
        for i in range(df.shape[1]):
            df.iloc[:, i] = df.iloc[:, i].apply(encod)
        # print df
        df.to_csv(self.typeSign[fileType][1], sep=",", index=False,encoding='gb18030')
        self.txtFileList.append(self.typeSign[fileType][1])
        print "sink to File success!"

    def zipFile(self,zipFilePath):


        def zipF(zipFilePath,filePath):
            if os.path.isfile(zipFilePath):  # abc.zip isfile exists ->a  else ->'w'
                zipf = zipfile.ZipFile(zipFilePath, 'a', zipfile.ZIP_STORED)
            else:
                zipf = zipfile.ZipFile(zipFilePath, 'w', zipfile.ZIP_STORED)
            zipf.write(filePath)
            zipf.close()

        for i in range(len(self.txtFileList)):
            filePath=self.txtFileList.pop()
            zipF(zipFilePath,filePath)
        print 'zip %s success!' % filePath


    def generatorZipFileName(self,fileSign, fileType='债券融资'):  # 指定文件类型 可选项：'债券融资'，'删除'; fileSign 0001-9999流水号。需要从数据库中获取
        orgSign = '123123123'
        timeFormate = '%Y%m%d%H%M'
        timeSign = time.strftime(timeFormate, time.localtime())
        # zfill填充为0001 4位数字
        fileSign = str(fileSign)
        fileSign = fileSign.zfill(4)
        fileName = orgSign + timeSign + self.typeSign[fileType][0] + fileSign
        if len(fileName) != 27:
            print 'File Name not enough 27 chars long!'
            exit(1)
        return fileName + '.zip',fileName+'.enc'

    def pading(self,text):
        length = 16
        count = len(text)
        print count
        if count < length:
            add = length - count
            text = text + ('\0' * add)

        elif count > length:
            add = (length - (count % length))
            text = text + ('\0' * add)
        return text


    def sftpF(self,filepath, ftpconf=''):
        if not os.path.isfile(ftpconf):
            ftpconf = os.path.split(args[0])[0] + '/ftpConf.ini'
        confPsr = ConfigParser.ConfigParser()
        confPsr.read(ftpconf)
        HOST = confPsr.get('base_info', 'HOST')
        PORT = confPsr.get('base_info', 'PORT')
        USER = confPsr.get('base_info', 'USER')
        PASSWD = confPsr.get('base_info', 'PASSWD')
        DIRN = confPsr.get('base_info', 'DIRN')
        # print DIRN,PASSWD
        if not os.path.isfile(filepath):
            print 'file %s not exists' % filepath
            exit(1)
        FILE = filepath
        transport = paramiko.Transport(HOST, int(PORT))
        transport.connect(username=USER, password=PASSWD)
        sftp = paramiko.SFTPClient.from_transport(transport)
        # filelist = sftp.listdir(DIRN)
        # 将location.py 上传至服务器 /tmp/test.py
        sftp.put(FILE, DIRN + "/" + FILE)
        # 将remove_path 下载到本地 local_path
        # sftp.get('/root/oldgirl.txt', 'fromlinux.txt')

        transport.close()

    """
        单次加密串的长度最大为 (key_size/8)-11
        1024bit的证书用100， 2048bit的证书用 200
    """

    def enc(self,srcFilePath,encMsgPath,enc_length=200):
        if not os.path.isfile(srcFilePath):
            raise Exception('File Not Found %s'%srcFilePath)


        with open(srcFilePath) as f:
            message = f.read()
        encMsg=''
        with open('master-public.pem','rb') as f:
            key = f.read()
            rsakey = RSA.importKey(key)
            cipher = Cipher_pkcs1_v1_5.new(rsakey)
            res = []
            for i in range(0, len(message), enc_length):
                res.append(cipher.encrypt(message[i:i + enc_length]))
            encMsg=encMsg.join(res)
        with open(encMsgPath,'wb') as f:
            f.write(encMsg)

    """
        解密：1024bit的证书用128，2048bit证书用256位
        """
    def dec(self,encMsgPath,decMsgPath,dec_length=256):

        encMsg=''
        #将加密文件读取到encMsg字符串中
        with open(encMsgPath,'rb') as f:
            encMsg=f.read()

        decMsg=''

        with open('master-private.pem','r') as f:
            key = f.read()
            rsakey = RSA.importKey(key)
            cipher = Cipher_pkcs1_v1_5.new(rsakey)
            res = []
            for i in range(0, len(encMsg), dec_length):
                res.append(cipher.decrypt(encMsg[i:i + dec_length], 'finupgroup.com')) #'xyz'是随机数，这里也可以填写为random_generator = Random.new().read 伪随机数生成器
            decMsg=decMsg.join(res)
            #将解密后的字符串整体存入到目标目录
        with open(decMsgPath,'wb') as f:
            f.write(decMsg)




if __name__ == '__main__':
    args = sys.argv

    ftpCl=SftpZipedFile()
    sql="SELECT row_number() over (order by apply_no) as rowid,`电销拨打优先级` as label,* FROM  develop.qianzhan_noloan_trun_chedai_result t limit 100"
    df=ftpCl.readFromHive(sql)#hive to DF

    ftpCl.df2file(df)  #DF 2 File
    zipPath,encPath=ftpCl.generatorZipFileName(1)  #生成 ZipName 和 EncFileName
    print zipPath,encPath
    ftpCl.zipFile(zipPath)  #ZIP压缩 ZipName Path


    ftpCl.enc(zipPath,encPath) #将Zip包加密，并生成enc文件
    ftpCl.sftpF(encPath)#SFTP上传enc文件

    print '*'*40
    ftpCl.dec(encPath,'eee_dec_True2.zip')#解压Enc文件至特定目录，查看手动解压查看是否和原文件内容一致。
    print '*' * 40
    
