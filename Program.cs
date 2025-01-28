using System;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using static EIT_SOLVER.MainForm;

namespace EIT_SOLVER
{
    static class Program
    {
        [DllImport("kernel32.dll")]
        static extern bool AttachConsole(int dwProcessId);
        private const int ATTACH_PARENT_PROCESS = -1;

        [STAThread]
        static void Main(string[] args)
        {
            // redirect console output to parent process;
            // must be before any calls to Console.WriteLine()
            AttachConsole(ATTACH_PARENT_PROCESS);

            // to demonstrate where the console output is going
            int argCount = args == null ? 0 : args.Length;
            Console.WriteLine("nYou specified {0} arguments:", argCount);
            if(argCount > 0)
            {
                for (int i = 0; i < argCount; i++)
                {
                    if (args[i] != null)
                        Console.WriteLine("  {0}", args[i]);
                }
            }

            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Console.WriteLine("Starting Application!");
            Application.Run(new MainForm());
        }
    }
}