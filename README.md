#CollegeMatch

Match college is based on a simple bilinear model to make recommendations for high school students with their profiles, and colleges' multi-dimensional data. We expect that the system could help students to find the colleges that best match their preference.

Dependencies:

1. amqp-client-4.0.0.jar
2. commons-pool2-2.4.2.jar
3. jedis-2.9.0.jar

The training profiles include two parties: colleges and students. Both parties required to transform all the training data to numerical values. Their format are similar except their features and the dimensions.

Here are some simple examples to show what the trainning data looks like.

The input features of the collges should be preprocessed to the following format (csv):

\# college_index f_1 f_2 f_3 ... f_dc

c001 0.3 0.8 0.0 ... 1.2

c002 0.9 0.2 0.7 ... 0.1

c003 1.2 0.0 0.0 ... 0.4

.   .   .   .   .   .

c990 0.9 0.1 0.4 ... 0.2

The input features of the students should be preprocessed into the similar format (csv):

\# student_index f_1 f_2 f_3 ... f_ds

s001 1.3 18 1.0 ... 0.2

s002 2.9 20 0.3 ... 1.4

s003 3.2 09 0.8 ... 0.5

.   .   .   .   .   .

s999 0.9 10 0.0 ... 1.2

Moreover, the desired rankings of students over collges should be collected based on survey result. These preference rankings are the ground-truth labels which are the required answers in supervised machine learning. These rankings should be associated to the corresponding students' index. Here is the specification for the preference rankings:

\# student_index college_first college_second ...

s020 c009 c001 c999

s029 c008 c002 c001 c909

s038 c287 c678 c21

s056 c629 c001 c200 c871 c323 c111 c002

The preference rankings over colleges provided by students are allowed different size. Some of them could just provide only two collges that (s)he most likes or dislikes. Also, (s)he can give more detailed feedback on his or her preferences over known colleges with a list of size 15. 


Based on these training data, CollegeMatch will be able to train a bilinear model by transforming both the original college features and the students features to a smaller feature space. The matchness between a student and a college could be computed in this reduced dimension. Each pair of student-college will be assigned a real score, and colleges could be ranked based on these scores. The higher a student-college's score is, the higher the probability that the student likes the college in the pair.

The training algorithm is under-development, and will support much advanced functionalities and produces more accurate results with increasing numbers of high-quality training data provided by students.
