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
        private int profile;

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

            label3.Parent = panel1;
            label3.BackColor = Color.Transparent;
            label3.Refresh();

            label4.Parent = panel2;
            label4.BackColor = Color.Transparent;
            label4.Refresh();
            label4.AutoSize = false;
            label4.TextAlign = ContentAlignment.TopCenter;
            label4.Dock = DockStyle.Fill;
            label4.Text = "Easy";

            panel1.Parent = tabPage2;
            panel1.BackColor = Color.Transparent;
            panel1.Refresh();
                
            music = true;
            profile = 0;

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
            Clean();
            tabControl1.SelectTab(2);
            rotatedLabelCS1.Text = "Score: 0";

            inputFile = File.Create("input.txt");
            inputFile.Close();
            File.WriteAllText("input.txt", profile.ToString());
            inputFile.Close();

            // open benchmarking programs...
            Process.Start("vectorAdd.exe");
            timer1.Start(); // we're just mocking the opening of a program
        }

        private void button6_Click(object sender, EventArgs e)
        {
            // stop programs and stuff
            tabControl1.SelectTab(0);
            timer1.Stop();
        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            // do all this stuff if the file is created
            if (File.Exists("results.txt"))
            {
                string score = File.ReadAllText("results.txt");
                rotatedLabelCS1.Text = "Score: " + score;

                if (File.Exists("input.txt"))
                    File.Delete("input.txt");

                tabControl1.SelectTab(3);
                timer1.Stop();
            }
        }

        private void button7_Click(object sender, EventArgs e)
        {
            tabControl1.SelectTab(0);
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

        private void macTrackBar2_MouseDown(object sender, MouseEventArgs e)
        {
            macTrackBar2.TrackerColor = Color.OrangeRed;
        }

        private void macTrackBar2_MouseUp(object sender, MouseEventArgs e)
        {
            macTrackBar2.TrackerColor = Color.DarkOrange;
        }

        private void macTrackBar2_ValueChanged(object sender, decimal value)
        {
            if (macTrackBar2.Value == 0)
                label4.Text = "Easy";
            else if (macTrackBar2.Value == 1)
                label4.Text = "Medium";
            else if (macTrackBar2.Value == 2)
                label4.Text = "Hard";
            else if (macTrackBar2.Value == 3)
                label4.Text = "Extreme";
            profile = macTrackBar2.Value;
        }
    }
}
