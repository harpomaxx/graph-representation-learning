from optparse import OptionParser
import sys

def parsearg():
    parser = OptionParser(usage="usage: %prog [opt] ",
                          version="%prog 1.0")
    parser.add_option("-i", "--input",
                      action="store",
                      dest="input",
                      default=False,
                      help="set name of the input file")
    parser.add_option("-o", "--output",
                      action="store", # optional because action defaults to "store"
                      dest="output",
                      help="set name of the output file",)
    (opt, args) = parser.parse_args()
    return opt


if __name__ == '__main__':
   opt = parsearg()
   if opt.input == None or opt.output == None:
        print("[Py] Parameters missing. Please use --help for look at available parameters.")
        sys.exit()

   else:
       print(opt.input)
       print(opt.output)
       if(opt.input == "holanda"):
              print("holanda es correcta entrada pa")
       #ncol_file  = pd.read_csv(opt.input, sep=' ', header=None)
       #pd.read_csv receives 
       
    #    output_df.to_csv(opt.output)