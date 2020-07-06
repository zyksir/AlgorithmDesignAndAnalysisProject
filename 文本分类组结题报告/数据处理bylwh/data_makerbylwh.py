import os
import json
filenames=os.listdir(r'mydata')
with open('mydata.json','w') as mydata:
    for filename in filenames:
        pathname='mydata/'+filename
        try:
            with open(pathname) as file_object:
                lines=file_object.readlines()
            comments=''
            for line in lines[14:]:
                comments+=line.rstrip()
            mydic={}
            if int(filename)<40063:
                mydic['type']='1'
            elif int(filename)<54564:
                mydic['type']='2'
            elif int(filename)<68339:
                mydic['type']='3'
            elif int(filename)<84571:
                mydic['type']='4'
            else:
                mydic['type']='5'
            mydic['text']=comments
            json.dump(mydic,mydata)
            mydata.write('\n')
        except UnicodeDecodeError:
            pass
        
