from Read_File import File_Handlers

if __name__ == "__main__":
     File = File_Handlers("Sample.txt")
     ans = File.Read_File()
     print(ans)