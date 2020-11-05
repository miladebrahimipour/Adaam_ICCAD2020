module s344 (
B0,
A1,
B1,
A2,
A3,
blif_clk_net,
START,
B3,
A0,
blif_reset_net,
B2,
P7,
P5,
CNTVCON2,
P2,
CNTVCO2,
P1,
P0,
READY,
P6,
P3,
P4);

// Start PIs
input B0;
input A1;
input B1;
input A2;
input A3;
input blif_clk_net;
input START;
input B3;
input A0;
input blif_reset_net;
input B2;

// Start POs
output P7;
output P5;
output CNTVCON2;
output P2;
output CNTVCO2;
output P1;
output P0;
output READY;
output P6;
output P3;
output P4;

// Start wires
wire net_166;
wire net_107;
wire net_47;
wire net_159;
wire net_61;
wire net_137;
wire net_132;
wire net_54;
wire net_105;
wire net_62;
wire P3;
wire net_6;
wire net_129;
wire net_119;
wire net_98;
wire net_23;
wire P5;
wire net_117;
wire net_12;
wire B1;
wire net_151;
wire net_74;
wire net_53;
wire net_93;
wire net_168;
wire net_135;
wire net_130;
wire net_147;
wire net_127;
wire net_14;
wire P1;
wire net_113;
wire net_26;
wire net_76;
wire blif_clk_net;
wire net_101;
wire net_32;
wire net_111;
wire net_90;
wire net_40;
wire net_100;
wire net_85;
wire net_69;
wire net_124;
wire net_161;
wire net_141;
wire net_160;
wire net_83;
wire net_115;
wire B3;
wire net_4;
wire net_95;
wire net_17;
wire net_78;
wire A1;
wire net_27;
wire net_164;
wire net_56;
wire net_87;
wire net_0;
wire net_155;
wire net_35;
wire net_16;
wire net_22;
wire net_39;
wire net_157;
wire net_144;
wire net_102;
wire net_2;
wire net_59;
wire net_9;
wire net_42;
wire net_120;
wire A3;
wire net_109;
wire net_80;
wire net_65;
wire blif_reset_net;
wire net_50;
wire net_162;
wire net_96;
wire net_66;
wire net_38;
wire net_44;
wire net_167;
wire net_136;
wire net_134;
wire net_19;
wire net_89;
wire net_45;
wire net_126;
wire B0;
wire net_34;
wire net_108;
wire net_150;
wire net_63;
wire P2;
wire net_152;
wire net_116;
wire net_30;
wire net_91;
wire net_106;
wire net_24;
wire net_55;
wire net_99;
wire net_46;
wire net_140;
wire net_118;
wire P7;
wire net_148;
wire net_104;
wire net_146;
wire net_72;
wire net_122;
wire net_25;
wire net_7;
wire net_70;
wire P4;
wire net_5;
wire net_52;
wire net_165;
wire net_128;
wire P0;
wire net_138;
wire net_13;
wire P6;
wire net_94;
wire net_11;
wire CNTVCON2;
wire net_18;
wire net_123;
wire net_131;
wire net_114;
wire CNTVCO2;
wire net_170;
wire net_29;
wire net_68;
wire net_149;
wire net_142;
wire net_77;
wire net_20;
wire net_31;
wire net_36;
wire net_49;
wire net_158;
wire net_15;
wire net_41;
wire net_57;
wire A2;
wire net_71;
wire net_153;
wire START;
wire net_156;
wire net_3;
wire net_84;
wire net_154;
wire net_112;
wire net_1;
wire net_92;
wire net_103;
wire net_139;
wire net_43;
wire net_10;
wire net_28;
wire net_169;
wire net_21;
wire net_51;
wire net_79;
wire net_143;
wire net_97;
wire net_88;
wire net_145;
wire net_60;
wire net_81;
wire net_163;
wire net_58;
wire B2;
wire net_67;
wire net_82;
wire net_64;
wire net_37;
wire net_110;
wire net_121;
wire net_73;
wire net_33;
wire net_48;
wire net_8;
wire net_75;
wire net_86;
wire net_133;
wire READY;
wire A0;
wire net_125;

// Start cells
CLKBUF_X2 inst_145 ( .A(net_133), .Z(net_134) );
INV_X2 inst_103 ( .A(net_22), .ZN(net_17) );
DFFR_X1 inst_125 ( .D(net_97), .RN(net_96), .QN(net_2), .CK(net_138) );
CLKBUF_X2 inst_138 ( .A(net_126), .Z(net_127) );
CLKBUF_X2 inst_159 ( .A(net_138), .Z(net_148) );
NAND3_X2 inst_15 ( .ZN(net_84), .A3(net_78), .A1(net_59), .A2(P6) );
CLKBUF_X2 inst_134 ( .A(net_122), .Z(net_123) );
CLKBUF_X2 inst_179 ( .A(net_148), .Z(net_168) );
NAND3_X1 inst_24 ( .ZN(net_83), .A3(net_82), .A1(net_73), .A2(net_72) );
DFFR_X2 inst_114 ( .RN(net_96), .D(net_69), .QN(net_5), .CK(net_160) );
OR2_X4 inst_6 ( .ZN(net_41), .A2(READY), .A1(B1) );
CLKBUF_X2 inst_131 ( .A(net_119), .Z(net_120) );
INV_X4 inst_76 ( .ZN(net_19), .A(net_10) );
CLKBUF_X2 inst_180 ( .A(net_168), .Z(net_169) );
CLKBUF_X2 inst_160 ( .A(net_119), .Z(net_149) );
CLKBUF_X2 inst_150 ( .A(net_129), .Z(net_139) );
NAND2_X4 inst_33 ( .A2(net_104), .A1(net_103), .ZN(net_90) );
CLKBUF_X2 inst_172 ( .A(net_133), .Z(net_161) );
INV_X4 inst_83 ( .ZN(net_32), .A(net_11) );
NAND2_X2 inst_47 ( .ZN(net_50), .A1(net_34), .A2(net_31) );
NAND3_X2 inst_19 ( .ZN(net_117), .A2(net_91), .A3(net_87), .A1(net_84) );
DFFR_X2 inst_123 ( .RN(net_96), .D(net_93), .QN(net_1), .CK(net_128) );
DFFR_X2 inst_121 ( .RN(net_96), .D(net_79), .QN(net_13), .CK(net_134) );
OR3_X2 inst_2 ( .A3(net_113), .A1(net_112), .ZN(net_65), .A2(net_56) );
NOR2_X2 inst_8 ( .A2(net_29), .A1(net_14), .ZN(CNTVCO2) );
DFFR_X2 inst_118 ( .RN(net_96), .D(net_76), .QN(net_11), .CK(net_147) );
INV_X4 inst_86 ( .A(net_73), .ZN(P6) );
CLKBUF_X2 inst_153 ( .A(net_141), .Z(net_142) );
NAND3_X2 inst_20 ( .A2(net_117), .A3(net_94), .ZN(net_93), .A1(net_57) );
NAND2_X4 inst_27 ( .ZN(net_62), .A1(net_30), .A2(net_27) );
NAND2_X2 inst_38 ( .ZN(net_45), .A2(net_30), .A1(net_17) );
INV_X2 inst_100 ( .A(net_48), .ZN(P4) );
NAND2_X2 inst_52 ( .A2(net_62), .ZN(net_58), .A1(net_48) );
INV_X4 inst_90 ( .ZN(net_44), .A(net_24) );
CLKBUF_X2 inst_140 ( .A(net_122), .Z(net_129) );
NAND2_X2 inst_40 ( .ZN(net_33), .A1(net_32), .A2(READY) );
CLKBUF_X2 inst_162 ( .A(net_150), .Z(net_151) );
CLKBUF_X2 inst_167 ( .A(net_124), .Z(net_156) );
INV_X2 inst_93 ( .A(net_114), .ZN(net_113) );
INV_X4 inst_81 ( .ZN(net_28), .A(net_4) );
INV_X2 inst_95 ( .ZN(net_26), .A(net_7) );
XOR2_X2 inst_1 ( .Z(net_63), .B(net_45), .A(net_16) );
INV_X4 inst_72 ( .ZN(net_115), .A(P0) );
CLKBUF_X2 inst_139 ( .A(net_127), .Z(net_128) );
CLKBUF_X2 inst_155 ( .A(net_123), .Z(net_144) );
NAND2_X2 inst_59 ( .ZN(net_75), .A1(net_62), .A2(net_50) );
CLKBUF_X2 inst_135 ( .A(net_123), .Z(net_124) );
NAND2_X2 inst_44 ( .A1(net_113), .ZN(net_99), .A2(net_56) );
NAND2_X2 inst_55 ( .ZN(net_111), .A1(net_44), .A2(net_2) );
CLKBUF_X2 inst_174 ( .A(net_162), .Z(net_163) );
DFFR_X2 inst_115 ( .RN(net_96), .D(net_70), .QN(net_4), .CK(net_155) );
NAND2_X2 inst_37 ( .A2(net_116), .ZN(net_47), .A1(net_26) );
CLKBUF_X2 inst_148 ( .A(net_136), .Z(net_137) );
CLKBUF_X2 inst_164 ( .A(net_152), .Z(net_153) );
OR2_X4 inst_5 ( .ZN(net_40), .A2(READY), .A1(B2) );
CLKBUF_X2 inst_157 ( .A(net_145), .Z(net_146) );
INV_X4 inst_84 ( .ZN(net_37), .A(net_12) );
NAND2_X2 inst_51 ( .A2(net_62), .ZN(net_57), .A1(net_56) );
CLKBUF_X2 inst_142 ( .A(net_120), .Z(net_131) );
INV_X4 inst_80 ( .ZN(net_20), .A(net_5) );
CLKBUF_X2 inst_173 ( .A(net_161), .Z(net_162) );
INV_X2 inst_105 ( .A(CNTVCO2), .ZN(CNTVCON2) );
MUX2_X2 inst_68 ( .Z(net_70), .B(net_28), .S(net_27), .A(A0) );
INV_X4 inst_78 ( .ZN(net_73), .A(net_2) );
NAND2_X2 inst_42 ( .ZN(net_38), .A1(net_37), .A2(READY) );
CLKBUF_X2 inst_175 ( .A(net_163), .Z(net_164) );
NAND2_X2 inst_53 ( .A2(net_62), .ZN(net_61), .A1(net_60) );
CLKBUF_X2 inst_177 ( .A(net_164), .Z(net_166) );
CLKBUF_X2 inst_133 ( .A(net_121), .Z(net_122) );
NAND2_X4 inst_26 ( .ZN(net_112), .A1(net_100), .A2(net_28) );
CLKBUF_X2 inst_151 ( .A(net_139), .Z(net_140) );
DFFR_X2 inst_112 ( .RN(net_96), .D(net_55), .QN(net_10), .CK(net_167) );
NAND2_X1 inst_64 ( .A2(net_88), .ZN(net_87), .A1(net_83) );
INV_X2 inst_107 ( .ZN(net_110), .A(net_64) );
MUX2_X2 inst_67 ( .Z(net_69), .S(net_27), .B(net_20), .A(A1) );
CLKBUF_X2 inst_181 ( .A(net_169), .Z(net_170) );
AND2_X4 inst_127 ( .A1(net_116), .ZN(net_100), .A2(net_0) );
MUX2_X2 inst_70 ( .S(net_91), .Z(net_77), .A(net_52), .B(net_35) );
CLKBUF_X2 inst_129 ( .A(blif_clk_net), .Z(net_118) );
INV_X4 inst_92 ( .ZN(net_91), .A(net_62) );
NAND2_X4 inst_29 ( .A2(net_99), .A1(net_98), .ZN(net_82) );
NAND3_X2 inst_17 ( .ZN(net_109), .A1(net_88), .A3(net_60), .A2(net_47) );
NOR2_X2 inst_11 ( .ZN(net_74), .A2(net_63), .A1(START) );
CLKBUF_X2 inst_146 ( .A(net_126), .Z(net_135) );
NAND3_X2 inst_14 ( .A1(net_113), .A3(net_112), .A2(net_56), .ZN(net_54) );
DFFR_X2 inst_122 ( .RN(net_96), .D(net_89), .QN(net_0), .CK(net_130) );
NAND2_X4 inst_31 ( .A2(net_102), .A1(net_101), .ZN(net_88) );
NAND2_X4 inst_25 ( .A2(net_116), .ZN(net_21), .A1(net_20) );
CLKBUF_X3 inst_126 ( .A(net_116), .Z(P0) );
CLKBUF_X2 inst_158 ( .A(net_146), .Z(net_147) );
CLKBUF_X2 inst_141 ( .A(net_129), .Z(net_130) );
NAND2_X2 inst_62 ( .ZN(net_92), .A1(net_91), .A2(net_90) );
INV_X1 inst_110 ( .A(net_82), .ZN(net_78) );
INV_X4 inst_74 ( .ZN(net_56), .A(net_1) );
NAND2_X2 inst_57 ( .A2(net_91), .ZN(net_71), .A1(net_32) );
NAND2_X2 inst_35 ( .A2(net_116), .ZN(net_24), .A1(net_23) );
INV_X2 inst_99 ( .A(net_60), .ZN(P7) );
NAND2_X2 inst_48 ( .ZN(net_51), .A1(net_39), .A2(net_36) );
MUX2_X2 inst_69 ( .S(net_91), .Z(net_76), .A(net_53), .B(net_37) );
NAND2_X2 inst_46 ( .ZN(net_49), .A1(net_48), .A2(net_25) );
INV_X4 inst_82 ( .ZN(net_60), .A(net_3) );
CLKBUF_X2 inst_136 ( .A(net_124), .Z(net_125) );
NAND2_X4 inst_30 ( .A2(net_111), .ZN(net_101), .A1(net_82) );
INV_X2 inst_102 ( .ZN(net_16), .A(net_9) );
INV_X2 inst_108 ( .A(net_88), .ZN(net_85) );
CLKBUF_X2 inst_165 ( .A(net_153), .Z(net_154) );
NAND2_X4 inst_32 ( .A2(net_110), .ZN(net_103), .A1(net_88) );
NAND3_X2 inst_22 ( .ZN(net_95), .A2(net_94), .A3(net_92), .A1(net_61) );
CLKBUF_X2 inst_144 ( .A(net_132), .Z(net_133) );
NAND2_X4 inst_34 ( .A2(net_109), .ZN(net_105), .A1(net_90) );
NAND3_X4 inst_12 ( .ZN(net_30), .A2(net_19), .A3(net_15), .A1(net_9) );
NAND2_X2 inst_56 ( .A2(net_112), .ZN(net_66), .A1(net_49) );
MUX2_X2 inst_71 ( .S(net_91), .Z(net_79), .B(net_66), .A(net_51) );
NAND3_X2 inst_21 ( .ZN(net_107), .A3(net_106), .A1(net_105), .A2(net_91) );
INV_X2 inst_104 ( .ZN(net_18), .A(P6) );
NAND2_X2 inst_60 ( .ZN(net_80), .A2(net_75), .A1(net_71) );
CLKBUF_X2 inst_169 ( .A(net_157), .Z(net_158) );
CLKBUF_X2 inst_168 ( .A(net_156), .Z(net_157) );
INV_X2 inst_97 ( .A(net_37), .ZN(P2) );
CLKBUF_X2 inst_161 ( .A(net_149), .Z(net_150) );
DFFR_X2 inst_124 ( .RN(net_96), .D(net_95), .QN(net_3), .CK(net_125) );
NAND3_X2 inst_18 ( .A2(net_94), .ZN(net_89), .A3(net_86), .A1(net_58) );
NAND3_X2 inst_16 ( .A2(net_91), .ZN(net_86), .A3(net_81), .A1(net_65) );
INV_X4 inst_88 ( .ZN(net_22), .A(net_15) );
OR2_X4 inst_3 ( .ZN(net_29), .A1(net_22), .A2(net_9) );
CLKBUF_X2 inst_156 ( .A(net_144), .Z(net_145) );
NOR2_X2 inst_9 ( .ZN(net_64), .A1(net_60), .A2(net_47) );
DFFR_X2 inst_113 ( .RN(net_96), .D(net_67), .QN(net_7), .CK(net_165) );
CLKBUF_X2 inst_170 ( .A(net_158), .Z(net_159) );
NAND2_X2 inst_50 ( .ZN(net_53), .A1(net_41), .A2(net_33) );
CLKBUF_X2 inst_137 ( .A(net_118), .Z(net_126) );
NAND2_X2 inst_41 ( .ZN(net_36), .A1(net_35), .A2(READY) );
CLKBUF_X2 inst_130 ( .A(net_118), .Z(net_119) );
INV_X4 inst_91 ( .ZN(net_72), .A(net_44) );
CLKBUF_X2 inst_132 ( .A(blif_clk_net), .Z(net_121) );
CLKBUF_X2 inst_143 ( .A(net_131), .Z(net_132) );
CLKBUF_X2 inst_176 ( .A(net_164), .Z(net_165) );
CLKBUF_X2 inst_152 ( .A(net_140), .Z(net_141) );
NAND2_X2 inst_58 ( .ZN(net_102), .A1(net_73), .A2(net_72) );
NAND2_X2 inst_36 ( .A2(net_116), .A1(net_28), .ZN(net_25) );
CLKBUF_X2 inst_147 ( .A(net_135), .Z(net_136) );
INV_X4 inst_87 ( .A(net_19), .ZN(net_14) );
NAND2_X2 inst_61 ( .ZN(net_106), .A2(net_85), .A1(net_64) );
NAND2_X2 inst_45 ( .ZN(net_104), .A1(net_60), .A2(net_47) );
INV_X2 inst_96 ( .A(net_32), .ZN(P1) );
INV_X2 inst_101 ( .A(net_56), .ZN(P5) );
XOR2_X2 inst_0 ( .Z(net_43), .A(net_29), .B(net_19) );
NOR2_X2 inst_10 ( .ZN(net_55), .A2(net_43), .A1(START) );
OR2_X4 inst_4 ( .ZN(net_39), .A2(READY), .A1(B3) );
MUX2_X2 inst_65 ( .Z(net_67), .S(net_27), .B(net_26), .A(A3) );
CLKBUF_X2 inst_178 ( .A(net_166), .Z(net_167) );
INV_X4 inst_89 ( .A(net_30), .ZN(READY) );
NAND2_X4 inst_28 ( .A2(net_112), .ZN(net_98), .A1(net_42) );
DFFR_X2 inst_111 ( .RN(net_96), .D(net_46), .QN(net_8), .CK(net_170) );
MUX2_X2 inst_66 ( .Z(net_68), .S(net_27), .B(net_23), .A(A2) );
DFFR_X2 inst_117 ( .RN(net_96), .D(net_74), .QN(net_9), .CK(net_148) );
INV_X2 inst_98 ( .A(net_35), .ZN(P3) );
NAND2_X1 inst_63 ( .A2(net_82), .ZN(net_81), .A1(net_54) );
OR2_X2 inst_7 ( .ZN(net_34), .A2(READY), .A1(B0) );
NAND2_X2 inst_49 ( .ZN(net_52), .A1(net_40), .A2(net_38) );
DFFR_X2 inst_120 ( .QN(net_116), .RN(net_96), .D(net_80), .CK(net_120) );
CLKBUF_X2 inst_154 ( .A(net_142), .Z(net_143) );
NAND3_X2 inst_13 ( .ZN(net_27), .A2(net_22), .A3(net_10), .A1(net_9) );
DFFR_X2 inst_119 ( .RN(net_96), .D(net_77), .QN(net_12), .CK(net_143) );
INV_X4 inst_75 ( .ZN(net_15), .A(net_8) );
CLKBUF_X2 inst_166 ( .A(net_154), .Z(net_155) );
DFFR_X2 inst_116 ( .RN(net_96), .D(net_68), .QN(net_6), .CK(net_151) );
CLKBUF_X2 inst_163 ( .A(net_122), .Z(net_152) );
INV_X4 inst_85 ( .ZN(net_94), .A(START) );
NAND2_X2 inst_54 ( .ZN(net_108), .A2(net_62), .A1(net_18) );
INV_X4 inst_79 ( .ZN(net_35), .A(net_13) );
INV_X1 inst_109 ( .ZN(net_96), .A(blif_reset_net) );
INV_X2 inst_106 ( .A(net_72), .ZN(net_59) );
CLKBUF_X2 inst_149 ( .A(net_137), .Z(net_138) );
NAND2_X2 inst_43 ( .A2(net_114), .ZN(net_42), .A1(net_1) );
NAND2_X2 inst_39 ( .A1(net_115), .ZN(net_31), .A2(READY) );
AND2_X2 inst_128 ( .A1(net_94), .ZN(net_46), .A2(net_45) );
INV_X4 inst_73 ( .ZN(net_114), .A(net_21) );
NAND3_X2 inst_23 ( .A3(net_108), .A1(net_107), .ZN(net_97), .A2(net_94) );
CLKBUF_X2 inst_171 ( .A(net_159), .Z(net_160) );
INV_X4 inst_77 ( .ZN(net_48), .A(net_0) );
INV_X2 inst_94 ( .ZN(net_23), .A(net_6) );

endmodule
