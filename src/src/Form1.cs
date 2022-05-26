using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Media;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace src {

    public partial class Form1 : Form
    {
        private bool music;
        private System.Media.SoundPlayer snd = new System.Media.SoundPlayer((System.IO.Stream)Properties.Resources.song);
        private FileStream inputFile;
        private String selectedOption = null;
        private Process p;
        private int MemAllocSingleScore, MemAllocMultiScore, ProcPowerMultiScore;

        public Form1()
        {
            InitializeComponent();
        }
        protected override CreateParams CreateParams
        {
            get
            {
                CreateParams cp = base.CreateParams;
                cp.ExStyle |= 0x02000000;  // Turn on WS_EX_COMPOSITED
                return cp;
            }
        }
        
        private void Clean()
        {
            if (File.Exists("results.txt"))
                File.Delete("results.txt");
            if (File.Exists("input.txt"))
                File.Delete("input.txt");
            if (File.Exists("results_vec_add.txt"))
                File.Delete("results_vec_add.txt");
            if (File.Exists("results_matrix_mult.txt"))
                File.Delete("results_matrix_mult.txt");
        }

        private void ComputeStandardScore()
        {
            inputFile = File.Create("input.txt");
            inputFile.Close();

            File.WriteAllText("input.txt", "0");
            inputFile.Close();

            p = new Process();

            p.StartInfo = new ProcessStartInfo("vectorAdd.exe");
            p.StartInfo.CreateNoWindow = true;
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
            p.Start();

            ComputeMemAllocSingleScoreTimer.Start();
        }

        private void Button_Enter(object sender, EventArgs e)
        {
            Button btn = (Button)sender;
            btn.BackgroundImage = Properties.Resources.ButtonHover;
        }

        private void Button_Leave(object sender, EventArgs e)
        {
            Button btn = (Button)sender;
            btn.BackgroundImage = Properties.Resources.Button;
        }

        private void button4_Click(object sender, EventArgs e)
        {
            tabControl1.SelectTab(0);
        }

        private void Button_Up(object sender, MouseEventArgs e)
        {
            Button btn = (Button)sender;
            btn.BackgroundImage = Properties.Resources.Button;
        }

        private void Button_Down(object sender, MouseEventArgs e)
        {
            Button btn = (Button)sender;
            btn.BackgroundImage = Properties.Resources.ButtonDown;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            this.SetStyle(System.Windows.Forms.ControlStyles.AllPaintingInWmPaint, true);

            this.Icon = Properties.Resources.spongebob;

            this.TransparencyKey = Color.Turquoise;
            this.BackColor = Color.Turquoise;

            tabControl1.Appearance = TabAppearance.Buttons;
            tabControl1.ItemSize = new System.Drawing.Size(0, 1);
            tabControl1.Multiline = true;
            tabControl1.SizeMode = TabSizeMode.Fixed;

            pictureBox1.SizeMode = PictureBoxSizeMode.StretchImage;

            button6.Parent = pictureBox1;
            button6.BackColor = Color.Transparent;
            button6.Refresh();

            label1.Parent = pictureBox1;
            label1.BackColor = Color.Transparent;
            label1.Refresh();

            label2.Parent = tabPage4;
            label2.BackColor = Color.Transparent;
            label2.Refresh();

            rotatedLabelCS1.Parent = tabPage4;
            rotatedLabelCS1.BackColor = Color.Transparent;
            rotatedLabelCS1.Refresh();
            rotatedLabelCS1.Angle = 30;
            rotatedLabelCS1.AutoSize = true;

            // option labels
            optionHeader1.Parent = tabPage2;
            optionHeader1.BackColor = Color.Transparent;
            optionHeader1.Refresh();
            optionHeader1.Angle = 8;            
            
            optionHeader2.Parent = tabPage2;
            optionHeader2.BackColor = Color.Transparent;
            optionHeader2.Refresh();
            optionHeader2.Angle = 8;

            optionLabel1.Parent = tabPage2;
            optionLabel1.BackColor = Color.Transparent;
            optionLabel1.Refresh();
            optionLabel1.Angle = 8;

            optionLabel2.Parent = tabPage2;
            optionLabel2.BackColor = Color.Transparent;
            optionLabel2.Refresh();
            optionLabel2.Angle = 8;            
            
            optionLabel3.Parent = tabPage2;
            optionLabel3.BackColor = Color.Transparent;
            optionLabel3.Refresh();
            optionLabel3.Angle = 8;            
            
            optionLabel4.Parent = tabPage2;
            optionLabel4.BackColor = Color.Transparent;
            optionLabel4.Refresh();
            optionLabel4.Angle = 8;

            optionLabel5.Parent = tabPage2;
            optionLabel5.BackColor = Color.Transparent;
            optionLabel5.Refresh();
            optionLabel5.Angle = 8;

            button8.Parent = tabPage1;
            button8.BackColor = Color.Transparent;
            button8.Refresh();

            music = true;

            // play amazing Spungbob song
            snd.PlayLooping();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            tabControl1.SelectTab(1);
        }

        private void button5_Click(object sender, EventArgs e)
        {
            tabControl1.SelectTab(0);
        }

        private void button4_Click_1(object sender, EventArgs e)
        {
            if (selectedOption == null)
            {
                MessageBox.Show("You need to select an option!", "Benchmark");
            }
            else
            {
                Clean();
                tabControl1.SelectTab(2);

                if (selectedOption != "optionLabel5")
                {
                    rotatedLabelCS1.Text = "0 Krabby Patties";

                    inputFile = File.Create("input.txt");
                    inputFile.Close();

                    if (selectedOption == "optionLabel1" || selectedOption == "optionLabel3") // Single-Threaded
                        File.WriteAllText("input.txt", "0");
                    else
                        File.WriteAllText("input.txt", "1"); // Multi-Threaded
                    inputFile.Close();

                    // open benchmarking programs...
                    p = new Process();
                    if (selectedOption == "optionLabel1" || selectedOption == "optionLabel2")
                        p.StartInfo = new ProcessStartInfo("vectorAdd.exe");
                    else
                        p.StartInfo = new ProcessStartInfo("matrixMul.exe");
                    p.StartInfo.CreateNoWindow = true;
                    p.StartInfo.UseShellExecute = false;
                    p.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
                    p.Start();

                    timer1.Start();
                }
                else ComputeStandardScore();
            }
        }


        private void button6_Click(object sender, EventArgs e)
        {
            // stop programs and stuff
            p.Kill();
            p = null;
            tabControl1.SelectTab(0);
            timer1.Stop();
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            // do all this stuff if the file is created
            String outputName;
            if (selectedOption == "optionLabel1" || selectedOption == "optionLabel2")
                outputName = "results_vec_add.txt";
            else
                outputName = "results_matrix_mult.txt";

            if (File.Exists(outputName))
            {
                string score = File.ReadAllText(outputName);
                rotatedLabelCS1.Text = score + " Krabby Patties";
                tabControl1.SelectTab(3);

                timer1.Stop();
                p = null;
            }
        }

        private void button7_Click(object sender, EventArgs e)
        {
            tabControl1.SelectTab(0);
            Clean();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (!music)
            {
                music = true;
                button2.Text = "Music: Off";
                snd.PlayLooping();
            }
            else
            {
                music = false;
                button2.Text = "Music: On";
                snd.Stop();
            }
        }

        private void GreenMouseEnter(object sender, EventArgs e)
        {
            if (selectedOption != ((RotatedLabelCS)sender).Name)
                ((RotatedLabelCS)sender).ForeColor = Color.Lime;
        }

        private void WhiteMouseLeave(object sender, EventArgs e)
        {
            if (selectedOption != ((RotatedLabelCS)sender).Name)
                ((RotatedLabelCS)sender).ForeColor = Color.White;
        }

        private void SelectOption(object sender, EventArgs e)
        {
            selectedOption = ((RotatedLabelCS)sender).Name;

            ((RotatedLabelCS)(sender)).ForeColor = Color.LimeGreen;

            if (selectedOption != "optionLabel1")
                optionLabel1.ForeColor = Color.White;
            if (selectedOption != "optionLabel2")
                optionLabel2.ForeColor= Color.White;
            if (selectedOption != "optionLabel3")
                optionLabel3.ForeColor = Color.White;
            if (selectedOption != "optionLabel4")
                optionLabel4.ForeColor = Color.White;
            if (selectedOption != "optionLabel5")
                optionLabel5.ForeColor = Color.White;
        }

        private void Form1_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (p != null)
                p.Kill();
        }

        private void ComputeMemAllocMultiScoreTimer_Tick(object sender, EventArgs e)
        {
            // do all this stuff if the file is created
            if (File.Exists("results_vec_add.txt"))
            {
                string score = File.ReadAllText("results_vec_add.txt");
                MemAllocMultiScore = Int32.Parse(score);

                ComputeMemAllocMultiScoreTimer.Stop();
                p = null;

                Clean();

                inputFile = File.Create("input.txt");
                inputFile.Close();

                File.WriteAllText("input.txt", "1");
                inputFile.Close();

                p = new Process();

                p.StartInfo = new ProcessStartInfo("matrixMul.exe");
                p.StartInfo.CreateNoWindow = true;
                p.StartInfo.UseShellExecute = false;
                p.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
                p.Start();

                ComputeProcPowerMultiTimer.Start();
            }
        }

        private void ComputeProcPowerMultiTimer_Tick(object sender, EventArgs e)
        {
            // do all this stuff if the file is created
            if (File.Exists("results_matrix_mult.txt"))
            {
                string score = File.ReadAllText("results_matrix_mult.txt");
                ProcPowerMultiScore = Int32.Parse(score);

                ComputeMemAllocSingleScoreTimer.Stop();
                p = null;

                double StandardScore = (double)5000000000 / (MemAllocSingleScore * 7 + MemAllocMultiScore * 250 * 6 + ProcPowerMultiScore * 87 * 5);
                rotatedLabelCS1.Text = ((int)StandardScore).ToString() + " Krabby Patties";

                Clean();
                tabControl1.SelectTab(3);
            }
        }

        private void button8_Click(object sender, EventArgs e)
        {
            snd.Stop();
            tabControl1.SelectTab(4);
            snd = new System.Media.SoundPlayer((System.IO.Stream)Properties.Resources.siuu);
            snd.Play();
            SiuuTimer.Start();
        }

        private void SiuuTimer_Tick(object sender, EventArgs e)
        {
            snd.Stop();
            tabControl1.SelectTab(0);
            if (button2.Enabled == true)
                snd = new System.Media.SoundPlayer((System.IO.Stream)Properties.Resources.song);
            else snd = new System.Media.SoundPlayer((System.IO.Stream)Properties.Resources.imn);
            snd.PlayLooping();
            SiuuTimer.Stop();
        }

        private void pictureBox2_Click(object sender, EventArgs e)
        {
            snd = new System.Media.SoundPlayer((System.IO.Stream)Properties.Resources.imn);
            snd.PlayLooping();
            button2.BackgroundImage = Properties.Resources.ButtonDown;
            button2.Enabled = false;
        }

        private void ComputeMemAllocSingleScoreTimer_Tick(object sender, EventArgs e)
        {
            // do all this stuff if the file is created
            if (File.Exists("results_vec_add.txt"))
            {
                string score = File.ReadAllText("results_vec_add.txt");
                MemAllocSingleScore = Int32.Parse(score);

                ComputeMemAllocSingleScoreTimer.Stop();
                p = null;

                Clean();

                inputFile = File.Create("input.txt");
                inputFile.Close();

                File.WriteAllText("input.txt", "1");
                inputFile.Close();

                p = new Process();

                p.StartInfo = new ProcessStartInfo("vectorAdd.exe");
                p.StartInfo.CreateNoWindow = true;
                p.StartInfo.UseShellExecute = false;
                p.StartInfo.WindowStyle = ProcessWindowStyle.Hidden;
                p.Start();

                ComputeMemAllocMultiScoreTimer.Start();
            }
        }
    }
}
