namespace src
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.timer1 = new System.Windows.Forms.Timer(this.components);
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.label1 = new System.Windows.Forms.Label();
            this.button6 = new System.Windows.Forms.Button();
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.button8 = new System.Windows.Forms.Button();
            this.pictureBox2 = new System.Windows.Forms.PictureBox();
            this.button3 = new System.Windows.Forms.Button();
            this.button2 = new System.Windows.Forms.Button();
            this.button1 = new System.Windows.Forms.Button();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.optionLabel5 = new src.RotatedLabelCS();
            this.optionHeader2 = new src.RotatedLabelCS();
            this.optionLabel4 = new src.RotatedLabelCS();
            this.optionLabel3 = new src.RotatedLabelCS();
            this.optionHeader1 = new src.RotatedLabelCS();
            this.optionLabel2 = new src.RotatedLabelCS();
            this.optionLabel1 = new src.RotatedLabelCS();
            this.button5 = new System.Windows.Forms.Button();
            this.button4 = new System.Windows.Forms.Button();
            this.tabPage4 = new System.Windows.Forms.TabPage();
            this.rotatedLabelCS1 = new src.RotatedLabelCS();
            this.label2 = new System.Windows.Forms.Label();
            this.button7 = new System.Windows.Forms.Button();
            this.tabPage5 = new System.Windows.Forms.TabPage();
            this.ComputeMemAllocSingleScoreTimer = new System.Windows.Forms.Timer(this.components);
            this.ComputeMemAllocMultiScoreTimer = new System.Windows.Forms.Timer(this.components);
            this.ComputeProcPowerMultiTimer = new System.Windows.Forms.Timer(this.components);
            this.SiuuTimer = new System.Windows.Forms.Timer(this.components);
            this.tabPage3.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).BeginInit();
            this.tabPage2.SuspendLayout();
            this.tabPage4.SuspendLayout();
            this.SuspendLayout();
            // 
            // timer1
            // 
            this.timer1.Interval = 1000;
            this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
            // 
            // tabPage3
            // 
            this.tabPage3.Controls.Add(this.label1);
            this.tabPage3.Controls.Add(this.button6);
            this.tabPage3.Controls.Add(this.pictureBox1);
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(792, 424);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "tabPage3";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // label1
            // 
            this.label1.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.Color.Transparent;
            this.label1.FlatStyle = System.Windows.Forms.FlatStyle.Popup;
            this.label1.Font = new System.Drawing.Font("Comic Sans MS", 18F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.Color.White;
            this.label1.Location = new System.Drawing.Point(518, 347);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(262, 35);
            this.label1.TabIndex = 4;
            this.label1.Text = "Benchmark running...";
            // 
            // button6
            // 
            this.button6.BackColor = System.Drawing.Color.Transparent;
            this.button6.BackgroundImage = global::src.Properties.Resources.Button;
            this.button6.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.button6.FlatAppearance.BorderSize = 0;
            this.button6.FlatAppearance.MouseDownBackColor = System.Drawing.Color.Transparent;
            this.button6.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Transparent;
            this.button6.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.button6.Font = new System.Drawing.Font("Comic Sans MS", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button6.ForeColor = System.Drawing.Color.White;
            this.button6.Location = new System.Drawing.Point(22, 22);
            this.button6.Name = "button6";
            this.button6.Size = new System.Drawing.Size(118, 46);
            this.button6.TabIndex = 3;
            this.button6.Text = "Exit";
            this.button6.UseVisualStyleBackColor = false;
            this.button6.Click += new System.EventHandler(this.button6_Click);
            this.button6.MouseDown += new System.Windows.Forms.MouseEventHandler(this.Button_Down);
            this.button6.MouseEnter += new System.EventHandler(this.Button_Enter);
            this.button6.MouseLeave += new System.EventHandler(this.Button_Leave);
            this.button6.MouseUp += new System.Windows.Forms.MouseEventHandler(this.Button_Up);
            // 
            // pictureBox1
            // 
            this.pictureBox1.Dock = System.Windows.Forms.DockStyle.Fill;
            this.pictureBox1.Image = global::src.Properties.Resources.patrick_computer;
            this.pictureBox1.Location = new System.Drawing.Point(0, 0);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(792, 424);
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Controls.Add(this.tabPage4);
            this.tabControl1.Controls.Add(this.tabPage5);
            this.tabControl1.Location = new System.Drawing.Point(0, 0);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(800, 450);
            this.tabControl1.TabIndex = 0;
            // 
            // tabPage1
            // 
            this.tabPage1.BackgroundImage = global::src.Properties.Resources.main_bg;
            this.tabPage1.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.tabPage1.Controls.Add(this.button8);
            this.tabPage1.Controls.Add(this.pictureBox2);
            this.tabPage1.Controls.Add(this.button3);
            this.tabPage1.Controls.Add(this.button2);
            this.tabPage1.Controls.Add(this.button1);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(792, 424);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "tabPage1";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // button8
            // 
            this.button8.FlatAppearance.BorderSize = 0;
            this.button8.FlatAppearance.MouseDownBackColor = System.Drawing.Color.FromArgb(((int)(((byte)(192)))), ((int)(((byte)(0)))), ((int)(((byte)(0)))));
            this.button8.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Red;
            this.button8.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.button8.Location = new System.Drawing.Point(763, 6);
            this.button8.Name = "button8";
            this.button8.Size = new System.Drawing.Size(21, 20);
            this.button8.TabIndex = 4;
            this.button8.UseVisualStyleBackColor = true;
            this.button8.Click += new System.EventHandler(this.button8_Click);
            // 
            // pictureBox2
            // 
            this.pictureBox2.BackgroundImage = global::src.Properties.Resources._255px_Flag_of_Romania_svg;
            this.pictureBox2.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pictureBox2.Location = new System.Drawing.Point(21, 400);
            this.pictureBox2.Name = "pictureBox2";
            this.pictureBox2.Size = new System.Drawing.Size(28, 16);
            this.pictureBox2.TabIndex = 3;
            this.pictureBox2.TabStop = false;
            this.pictureBox2.Click += new System.EventHandler(this.pictureBox2_Click);
            // 
            // button3
            // 
            this.button3.BackColor = System.Drawing.Color.Transparent;
            this.button3.BackgroundImage = global::src.Properties.Resources.Button;
            this.button3.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.button3.FlatAppearance.BorderSize = 0;
            this.button3.FlatAppearance.MouseDownBackColor = System.Drawing.Color.Transparent;
            this.button3.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Transparent;
            this.button3.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.button3.Font = new System.Drawing.Font("Comic Sans MS", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button3.ForeColor = System.Drawing.Color.White;
            this.button3.Location = new System.Drawing.Point(411, 267);
            this.button3.Name = "button3";
            this.button3.Size = new System.Drawing.Size(177, 46);
            this.button3.TabIndex = 2;
            this.button3.Text = "Exit";
            this.button3.UseVisualStyleBackColor = false;
            this.button3.Click += new System.EventHandler(this.button3_Click);
            this.button3.MouseDown += new System.Windows.Forms.MouseEventHandler(this.Button_Down);
            this.button3.MouseEnter += new System.EventHandler(this.Button_Enter);
            this.button3.MouseLeave += new System.EventHandler(this.Button_Leave);
            this.button3.MouseUp += new System.Windows.Forms.MouseEventHandler(this.Button_Up);
            // 
            // button2
            // 
            this.button2.BackColor = System.Drawing.Color.Transparent;
            this.button2.BackgroundImage = global::src.Properties.Resources.Button;
            this.button2.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.button2.FlatAppearance.BorderSize = 0;
            this.button2.FlatAppearance.MouseDownBackColor = System.Drawing.Color.Transparent;
            this.button2.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Transparent;
            this.button2.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.button2.Font = new System.Drawing.Font("Comic Sans MS", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button2.ForeColor = System.Drawing.Color.White;
            this.button2.Location = new System.Drawing.Point(411, 183);
            this.button2.Name = "button2";
            this.button2.Size = new System.Drawing.Size(177, 46);
            this.button2.TabIndex = 1;
            this.button2.Text = "Music: Off";
            this.button2.UseVisualStyleBackColor = false;
            this.button2.Click += new System.EventHandler(this.button2_Click);
            this.button2.MouseDown += new System.Windows.Forms.MouseEventHandler(this.Button_Down);
            this.button2.MouseEnter += new System.EventHandler(this.Button_Enter);
            this.button2.MouseLeave += new System.EventHandler(this.Button_Leave);
            this.button2.MouseUp += new System.Windows.Forms.MouseEventHandler(this.Button_Up);
            // 
            // button1
            // 
            this.button1.BackColor = System.Drawing.Color.Transparent;
            this.button1.BackgroundImage = global::src.Properties.Resources.Button;
            this.button1.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.button1.FlatAppearance.BorderSize = 0;
            this.button1.FlatAppearance.MouseDownBackColor = System.Drawing.Color.Transparent;
            this.button1.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Transparent;
            this.button1.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.button1.Font = new System.Drawing.Font("Comic Sans MS", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button1.ForeColor = System.Drawing.Color.White;
            this.button1.Location = new System.Drawing.Point(411, 103);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(177, 46);
            this.button1.TabIndex = 0;
            this.button1.Text = "Benchmark";
            this.button1.UseVisualStyleBackColor = false;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            this.button1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.Button_Down);
            this.button1.MouseEnter += new System.EventHandler(this.Button_Enter);
            this.button1.MouseLeave += new System.EventHandler(this.Button_Leave);
            this.button1.MouseUp += new System.Windows.Forms.MouseEventHandler(this.Button_Up);
            // 
            // tabPage2
            // 
            this.tabPage2.BackgroundImage = global::src.Properties.Resources.WhatsApp_Image_2022_05_25_at_12_20_47_PM;
            this.tabPage2.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.tabPage2.Controls.Add(this.optionLabel5);
            this.tabPage2.Controls.Add(this.optionHeader2);
            this.tabPage2.Controls.Add(this.optionLabel4);
            this.tabPage2.Controls.Add(this.optionLabel3);
            this.tabPage2.Controls.Add(this.optionHeader1);
            this.tabPage2.Controls.Add(this.optionLabel2);
            this.tabPage2.Controls.Add(this.optionLabel1);
            this.tabPage2.Controls.Add(this.button5);
            this.tabPage2.Controls.Add(this.button4);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(792, 424);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "tabPage2";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // optionLabel5
            // 
            this.optionLabel5.Angle = 0;
            this.optionLabel5.AutoSize = true;
            this.optionLabel5.Font = new System.Drawing.Font("Consolas", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.optionLabel5.ForeColor = System.Drawing.Color.White;
            this.optionLabel5.Location = new System.Drawing.Point(562, 69);
            this.optionLabel5.Name = "optionLabel5";
            this.optionLabel5.Size = new System.Drawing.Size(176, 27);
            this.optionLabel5.TabIndex = 9;
            this.optionLabel5.Text = "Standard Testing";
            this.optionLabel5.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            this.optionLabel5.Click += new System.EventHandler(this.SelectOption);
            this.optionLabel5.MouseEnter += new System.EventHandler(this.GreenMouseEnter);
            this.optionLabel5.MouseLeave += new System.EventHandler(this.WhiteMouseLeave);
            // 
            // optionHeader2
            // 
            this.optionHeader2.Angle = 0;
            this.optionHeader2.AutoSize = true;
            this.optionHeader2.Font = new System.Drawing.Font("Consolas", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.optionHeader2.ForeColor = System.Drawing.Color.White;
            this.optionHeader2.Location = new System.Drawing.Point(537, 171);
            this.optionHeader2.Name = "optionHeader2";
            this.optionHeader2.Size = new System.Drawing.Size(176, 17);
            this.optionHeader2.TabIndex = 8;
            this.optionHeader2.Text = "Processing Power";
            this.optionHeader2.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            // 
            // optionLabel4
            // 
            this.optionLabel4.Angle = 0;
            this.optionLabel4.AutoSize = true;
            this.optionLabel4.Font = new System.Drawing.Font("Consolas", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.optionLabel4.ForeColor = System.Drawing.Color.White;
            this.optionLabel4.Location = new System.Drawing.Point(567, 225);
            this.optionLabel4.Name = "optionLabel4";
            this.optionLabel4.Size = new System.Drawing.Size(176, 27);
            this.optionLabel4.TabIndex = 7;
            this.optionLabel4.Text = "Multi-Threaded";
            this.optionLabel4.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            this.optionLabel4.Click += new System.EventHandler(this.SelectOption);
            this.optionLabel4.MouseEnter += new System.EventHandler(this.GreenMouseEnter);
            this.optionLabel4.MouseLeave += new System.EventHandler(this.WhiteMouseLeave);
            // 
            // optionLabel3
            // 
            this.optionLabel3.Angle = 0;
            this.optionLabel3.AutoSize = true;
            this.optionLabel3.Font = new System.Drawing.Font("Consolas", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.optionLabel3.ForeColor = System.Drawing.Color.White;
            this.optionLabel3.Location = new System.Drawing.Point(568, 201);
            this.optionLabel3.Name = "optionLabel3";
            this.optionLabel3.Size = new System.Drawing.Size(176, 27);
            this.optionLabel3.TabIndex = 6;
            this.optionLabel3.Text = "Single-Threaded";
            this.optionLabel3.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            this.optionLabel3.Click += new System.EventHandler(this.SelectOption);
            this.optionLabel3.MouseEnter += new System.EventHandler(this.GreenMouseEnter);
            this.optionLabel3.MouseLeave += new System.EventHandler(this.WhiteMouseLeave);
            // 
            // optionHeader1
            // 
            this.optionHeader1.Angle = 0;
            this.optionHeader1.AutoSize = true;
            this.optionHeader1.Font = new System.Drawing.Font("Consolas", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.optionHeader1.ForeColor = System.Drawing.Color.White;
            this.optionHeader1.Location = new System.Drawing.Point(556, 95);
            this.optionHeader1.Name = "optionHeader1";
            this.optionHeader1.Size = new System.Drawing.Size(176, 17);
            this.optionHeader1.TabIndex = 5;
            this.optionHeader1.Text = "Memory Allocation";
            this.optionHeader1.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            // 
            // optionLabel2
            // 
            this.optionLabel2.Angle = 0;
            this.optionLabel2.AutoSize = true;
            this.optionLabel2.Font = new System.Drawing.Font("Consolas", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.optionLabel2.ForeColor = System.Drawing.Color.White;
            this.optionLabel2.Location = new System.Drawing.Point(586, 149);
            this.optionLabel2.Name = "optionLabel2";
            this.optionLabel2.Size = new System.Drawing.Size(176, 27);
            this.optionLabel2.TabIndex = 4;
            this.optionLabel2.Text = "Multi-Threaded";
            this.optionLabel2.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            this.optionLabel2.Click += new System.EventHandler(this.SelectOption);
            this.optionLabel2.MouseEnter += new System.EventHandler(this.GreenMouseEnter);
            this.optionLabel2.MouseLeave += new System.EventHandler(this.WhiteMouseLeave);
            // 
            // optionLabel1
            // 
            this.optionLabel1.Angle = 0;
            this.optionLabel1.AutoSize = true;
            this.optionLabel1.Font = new System.Drawing.Font("Consolas", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.optionLabel1.ForeColor = System.Drawing.Color.White;
            this.optionLabel1.Location = new System.Drawing.Point(587, 125);
            this.optionLabel1.Name = "optionLabel1";
            this.optionLabel1.Size = new System.Drawing.Size(176, 27);
            this.optionLabel1.TabIndex = 3;
            this.optionLabel1.Text = "Single-Threaded";
            this.optionLabel1.TextAlign = System.Drawing.ContentAlignment.TopLeft;
            this.optionLabel1.Click += new System.EventHandler(this.SelectOption);
            this.optionLabel1.MouseEnter += new System.EventHandler(this.GreenMouseEnter);
            this.optionLabel1.MouseLeave += new System.EventHandler(this.WhiteMouseLeave);
            // 
            // button5
            // 
            this.button5.BackColor = System.Drawing.Color.Transparent;
            this.button5.BackgroundImage = global::src.Properties.Resources.Button;
            this.button5.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.button5.FlatAppearance.BorderSize = 0;
            this.button5.FlatAppearance.MouseDownBackColor = System.Drawing.Color.Transparent;
            this.button5.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Transparent;
            this.button5.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.button5.Font = new System.Drawing.Font("Comic Sans MS", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button5.ForeColor = System.Drawing.Color.White;
            this.button5.Location = new System.Drawing.Point(22, 22);
            this.button5.Name = "button5";
            this.button5.Size = new System.Drawing.Size(118, 46);
            this.button5.TabIndex = 2;
            this.button5.Text = "Exit";
            this.button5.UseVisualStyleBackColor = false;
            this.button5.Click += new System.EventHandler(this.button5_Click);
            this.button5.MouseDown += new System.Windows.Forms.MouseEventHandler(this.Button_Down);
            this.button5.MouseEnter += new System.EventHandler(this.Button_Enter);
            this.button5.MouseLeave += new System.EventHandler(this.Button_Leave);
            this.button5.MouseUp += new System.Windows.Forms.MouseEventHandler(this.Button_Up);
            // 
            // button4
            // 
            this.button4.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Right)));
            this.button4.BackColor = System.Drawing.Color.Transparent;
            this.button4.BackgroundImage = global::src.Properties.Resources.Button;
            this.button4.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.button4.FlatAppearance.BorderSize = 0;
            this.button4.FlatAppearance.MouseDownBackColor = System.Drawing.Color.Transparent;
            this.button4.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Transparent;
            this.button4.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.button4.Font = new System.Drawing.Font("Comic Sans MS", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button4.ForeColor = System.Drawing.Color.White;
            this.button4.Location = new System.Drawing.Point(528, 347);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(232, 46);
            this.button4.TabIndex = 1;
            this.button4.Text = "Start Benchmark";
            this.button4.UseVisualStyleBackColor = false;
            this.button4.Click += new System.EventHandler(this.button4_Click_1);
            this.button4.MouseDown += new System.Windows.Forms.MouseEventHandler(this.Button_Down);
            this.button4.MouseEnter += new System.EventHandler(this.Button_Enter);
            this.button4.MouseLeave += new System.EventHandler(this.Button_Leave);
            this.button4.MouseUp += new System.Windows.Forms.MouseEventHandler(this.Button_Up);
            // 
            // tabPage4
            // 
            this.tabPage4.BackgroundImage = global::src.Properties.Resources.results_bg;
            this.tabPage4.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.tabPage4.Controls.Add(this.rotatedLabelCS1);
            this.tabPage4.Controls.Add(this.label2);
            this.tabPage4.Controls.Add(this.button7);
            this.tabPage4.Location = new System.Drawing.Point(4, 22);
            this.tabPage4.Name = "tabPage4";
            this.tabPage4.Size = new System.Drawing.Size(792, 424);
            this.tabPage4.TabIndex = 3;
            this.tabPage4.Text = "tabPage4";
            this.tabPage4.UseVisualStyleBackColor = true;
            // 
            // rotatedLabelCS1
            // 
            this.rotatedLabelCS1.Angle = 0;
            this.rotatedLabelCS1.Font = new System.Drawing.Font("Comic Sans MS", 18F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.rotatedLabelCS1.Location = new System.Drawing.Point(295, 137);
            this.rotatedLabelCS1.Name = "rotatedLabelCS1";
            this.rotatedLabelCS1.Size = new System.Drawing.Size(384, 150);
            this.rotatedLabelCS1.TabIndex = 7;
            this.rotatedLabelCS1.Text = "1000 Krabby Patties";
            this.rotatedLabelCS1.TextAlign = System.Drawing.ContentAlignment.TopCenter;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Comic Sans MS", 21.75F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.ForeColor = System.Drawing.Color.White;
            this.label2.Location = new System.Drawing.Point(298, 366);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(219, 40);
            this.label2.TabIndex = 5;
            this.label2.Text = "Results are in!";
            // 
            // button7
            // 
            this.button7.BackColor = System.Drawing.Color.Transparent;
            this.button7.BackgroundImage = global::src.Properties.Resources.Button;
            this.button7.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.button7.FlatAppearance.BorderSize = 0;
            this.button7.FlatAppearance.MouseDownBackColor = System.Drawing.Color.Transparent;
            this.button7.FlatAppearance.MouseOverBackColor = System.Drawing.Color.Transparent;
            this.button7.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.button7.Font = new System.Drawing.Font("Comic Sans MS", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.button7.ForeColor = System.Drawing.Color.White;
            this.button7.Location = new System.Drawing.Point(22, 22);
            this.button7.Name = "button7";
            this.button7.Size = new System.Drawing.Size(239, 46);
            this.button7.TabIndex = 4;
            this.button7.Text = "Save and Exit";
            this.button7.UseVisualStyleBackColor = false;
            this.button7.Click += new System.EventHandler(this.button7_Click);
            this.button7.MouseDown += new System.Windows.Forms.MouseEventHandler(this.Button_Down);
            this.button7.MouseEnter += new System.EventHandler(this.Button_Enter);
            this.button7.MouseLeave += new System.EventHandler(this.Button_Leave);
            this.button7.MouseUp += new System.Windows.Forms.MouseEventHandler(this.Button_Up);
            // 
            // tabPage5
            // 
            this.tabPage5.BackgroundImage = global::src.Properties.Resources.maxresdefault;
            this.tabPage5.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.tabPage5.Location = new System.Drawing.Point(4, 22);
            this.tabPage5.Name = "tabPage5";
            this.tabPage5.Size = new System.Drawing.Size(792, 424);
            this.tabPage5.TabIndex = 4;
            this.tabPage5.Text = "tabPage5";
            this.tabPage5.UseVisualStyleBackColor = true;
            // 
            // ComputeMemAllocSingleScoreTimer
            // 
            this.ComputeMemAllocSingleScoreTimer.Tick += new System.EventHandler(this.ComputeMemAllocSingleScoreTimer_Tick);
            // 
            // ComputeMemAllocMultiScoreTimer
            // 
            this.ComputeMemAllocMultiScoreTimer.Tick += new System.EventHandler(this.ComputeMemAllocMultiScoreTimer_Tick);
            // 
            // ComputeProcPowerMultiTimer
            // 
            this.ComputeProcPowerMultiTimer.Tick += new System.EventHandler(this.ComputeProcPowerMultiTimer_Tick);
            // 
            // SiuuTimer
            // 
            this.SiuuTimer.Interval = 2000;
            this.SiuuTimer.Tick += new System.EventHandler(this.SiuuTimer_Tick);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(800, 450);
            this.Controls.Add(this.tabControl1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.MinimizeBox = false;
            this.Name = "Form1";
            this.Text = "Benchmark";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Form1_FormClosing);
            this.Load += new System.EventHandler(this.Form1_Load);
            this.tabPage3.ResumeLayout(false);
            this.tabPage3.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox2)).EndInit();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.tabPage4.ResumeLayout(false);
            this.tabPage4.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.TabPage tabPage4;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Button button7;
        private System.Windows.Forms.TabPage tabPage3;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Button button6;
        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.TabPage tabPage2;
        private System.Windows.Forms.Button button5;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.Button button3;
        private System.Windows.Forms.Button button2;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.TabControl tabControl1;
        private RotatedLabelCS rotatedLabelCS1;
        private System.Windows.Forms.Timer timer1;
        private RotatedLabelCS optionLabel1;
        private RotatedLabelCS optionHeader1;
        private RotatedLabelCS optionLabel2;
        private RotatedLabelCS optionHeader2;
        private RotatedLabelCS optionLabel4;
        private RotatedLabelCS optionLabel3;
        private System.Windows.Forms.PictureBox pictureBox2;
        private RotatedLabelCS optionLabel5;
        private System.Windows.Forms.Timer ComputeMemAllocSingleScoreTimer;
        private System.Windows.Forms.Timer ComputeMemAllocMultiScoreTimer;
        private System.Windows.Forms.Timer ComputeProcPowerMultiTimer;
        private System.Windows.Forms.Button button8;
        private System.Windows.Forms.Timer SiuuTimer;
        private System.Windows.Forms.TabPage tabPage5;
    }
}

