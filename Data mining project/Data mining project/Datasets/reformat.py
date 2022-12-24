x = open("diabetes.txt" )
s=x.read().replace(",", ";") 
x.close()

#Now open the file in write mode and write the string with replaced

x=open("diabetes.txt","w")
x.write(s)
x.close