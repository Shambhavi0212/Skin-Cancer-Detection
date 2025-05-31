/*
Reverse Characters Preserving Space Positions(spec)

Problem
You are given a sentence S that may contain multiple words separated by single spaces.
Your task is to reverse only the characters, while preserving the position of spaces in the original sentence.

That is, the characters should appear in reverse order from the original sentence, but the spaces should remain exactly where they were.

Input Format
A single line string S containing alphabets and spaces.

Constraints
1 ≤ S.length() ≤ 1000
The string contains only alphabets and single spaces.
No leading or trailing spaces.
No multiple consecutive spaces.

Output Format
A single line with characters reversed, but spaces retained at their original positions.

Sample Input 0
data structures are important

Sample Output 0
tnat ropmierase rut curtsatad

Sample Input 1
ravi is best in singing

Sample Output 1
gnig ni snit se bsiivar
*/
import java.io.*;
import java.util.*;

public class Solution {

    public static void main(String[] args) {
        Scanner sc=new Scanner(System.in);
        String s=sc.nextLine();
        
        String rs=s.replaceAll(" ","");
        int len=rs.length()-1;
        for(int i=0;i<s.length();i++)
        {
            if(s.charAt(i)==' ')
            {
                System.out.print(" ");
            }
            else
            {
                System.out.print(rs.charAt(len));
                len--;
            }
        }
        
        
    }
}