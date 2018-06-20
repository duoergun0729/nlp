# -*- coding: UTF-8 -*-
import itchat
import time

def demo():
    # 可以保持一段时间登录状态，而不用每次运行代码都要扫码登录了
    itchat.auto_login(hotReload=True)
    itchat.dump_login_status()

    #遍历好友列表
    friends = itchat.get_friends(update=True)[:]

    for friend in friends:
        '''
        典型属性内容如下：
        {'MemberList': <ContactList: []>, 'VerifyFlag': 0, 'DisplayName': '', 'EncryChatRoomId': '', 'Alias': '',
        'PYQuanPin': 'Evi1hui', 'PYInitial': 'EVI1HUI', 'RemarkName': '', 'AppAccountFlag': 0, 'City': '阳泉',
        'ChatRoomId': 0, 'AttrStatus': 2147715109, 'UniFriend': 0, 'OwnerUin': 0, 'Statues': 0, 'StarFriend': 0,
        'ContactFlag': 3,
        'HeadImgUrl': '/cgi-bin/mmwebwx-bin/webwxgeticon?seq=654224580&username=@78606d673dcff895a468688273bb5e92862c15d9a0e8a574865785c8a0354660&skey=@crypt_cd4d9fd6_f8d94dff59afaff6b231f3f4b8aa1e15',
        'Sex': 1, 'Uin': 0, 'HideInputBarFlag': 0, 'MemberCount': 0, 'Signature': '沉默的大多数',
        'NickName': 'Evi1hui', 'RemarkPYQuanPin': '',
        'UserName': '@78606d673dcff895a468688273bb5e92862c15d9a0e8a574865785c8a0354660',
        'IsOwner': 0, 'RemarkPYInitial': '', 'KeyWord': 'wan', 'Province': '山西', 'SnsFlag': 177}
        '''
        if friend['NickName'] == '兜哥的生活号':
            #print(friend['NickName'])
            print(friend)

    #发送的主键为UserName字段的值
    itchat.send_msg('Nice to meet you!','@1df002b437271f75f1afd07b937801b3a7211aad221e24115fb153674ca044b7')

def sendall(msg):
    itchat.auto_login(hotReload=True)
    itchat.dump_login_status()

    # 遍历好友列表
    friends = itchat.get_friends(update=True)[:]

    for friend in friends:
        '''
        典型属性内容如下：
        {'MemberList': <ContactList: []>, 'VerifyFlag': 0, 'DisplayName': '', 'EncryChatRoomId': '', 'Alias': '',
        'PYQuanPin': 'Evi1hui', 'PYInitial': 'EVI1HUI', 'RemarkName': '', 'AppAccountFlag': 0, 'City': '阳泉',
        'ChatRoomId': 0, 'AttrStatus': 2147715109, 'UniFriend': 0, 'OwnerUin': 0, 'Statues': 0, 'StarFriend': 0,
        'ContactFlag': 3,
        'HeadImgUrl': '/cgi-bin/mmwebwx-bin/webwxgeticon?seq=654224580&username=@78606d673dcff895a468688273bb5e92862c15d9a0e8a574865785c8a0354660&skey=@crypt_cd4d9fd6_f8d94dff59afaff6b231f3f4b8aa1e15',
        'Sex': 1, 'Uin': 0, 'HideInputBarFlag': 0, 'MemberCount': 0, 'Signature': '沉默的大多数',
        'NickName': 'Evi1hui', 'RemarkPYQuanPin': '',
        'UserName': '@78606d673dcff895a468688273bb5e92862c15d9a0e8a574865785c8a0354660',
        'IsOwner': 0, 'RemarkPYInitial': '', 'KeyWord': 'wan', 'Province': '山西', 'SnsFlag': 177}
        '''
        UserName=friend['UserName']
        NickName=friend['NickName']
        City=friend['City']

        #print("UserName:%s NickName:%s City:%s" % (UserName,NickName,City))
        # 发送的主键为UserName字段的值
        text="hi %s,%s" %(NickName,msg)

        # 休眠10秒
        time.sleep(10)
        print(text)
        itchat.send_msg(text, UserName)


if __name__ == '__main__':
    #demo()
    sendall('今天高温，记得防暑降温喔：）')
