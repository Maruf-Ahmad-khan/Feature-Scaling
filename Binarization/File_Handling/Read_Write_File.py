class Read_Write_Files():
     
     def __init__(self, FileName):
          self.FileName = FileName
          
          
     def Writes(self, contents):
          
          """
          Write something in that file.txt
          """
          with open(self.FileName, 'w') as file:
               file.write(contents)
               
               
     def Reads(self):
          
          """
          Read the text
          
          """
          
          with open(self.FileName, 'r') as file:
               return file.read()
          
          
if __name__ == "__main__":
     files = Read_Write_Files("file.txt")
     files.Writes("\nHello I am learning file handling in python")
     print("The contents in the file is \n",files.Reads())
               
                    
                    
                    