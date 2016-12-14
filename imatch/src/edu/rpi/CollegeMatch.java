package edu.rpi;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 *
 * @author Chunheng Jiang
 * @version 1.0
 * @since 11:52:26 AM, Nov 17, 2016
 */

public class CollegeMatch {
	public int maxIter = 200;
	public double eta = 1.0E-2;

	public CollegeMatch() {

	}

	public CollegeMatch setMaxIter(int maxIter) {
		this.maxIter = maxIter;
		return this;
	}

	public CollegeMatch setEta(double eta) {
		this.eta = eta;
		return this;
	}

	/**
	 * Learning a bilinear model based on the features of both colleges and
	 * students
	 * 
	 * @param colleges
	 *            college feature matrix
	 * @param students
	 *            stduent feature matrix
	 * @param preferences
	 *            students' preferences or ranking list over colleges, and each
	 *            integer array element presents a student's preference ranking
	 *            over some colleges. The ranking list for different students
	 *            allows various lengths.
	 * @return Tranformation matrix for both colleges and students
	 */
	public double[][] learn(double[][] colleges, double[][] students, List<int[]> preferences) {
		int ns = students.length, ds = students[0].length, dc = colleges[0].length;
		/**
		 * ws: ds -> d, wc: dc -> d, w = (ws)'(wc): dc->d->ds
		 */
		double[][] w = new double[ds][dc];
		for (int iter = 0; iter < maxIter; iter++) {
			int i = MathLib.Rand.sample(0, ns);
			double[][] g = gradient(students[i], preferences.get(i), colleges, w);
			w = MathLib.Matrix.subtract(w, MathLib.Matrix.multiply(g, -eta));
		}
		return w;
	}

	/**
	 * @param students
	 * @param colleges
	 * @param preferences
	 *            all students have the same size of preferences
	 * @return Transformation matrix for both colleges and students
	 */
	public double[][] learn(double[][] students, double[][] colleges, int[][] preferences) {
		int ns = students.length, ds = students[0].length, dc = colleges[0].length;

		double[][] w = new double[ds][dc];
		for (int iter = 0; iter < maxIter; iter++) {
			int i = MathLib.Rand.sample(0, ns);
			double[][] g = gradient(students[i], preferences[i], colleges, w);
			w = MathLib.Matrix.subtract(w, MathLib.Matrix.multiply(g, -eta));
		}
		return w;
	}

	/**
	 * Logistic function f(x) = 1.0/(1+e^{-x})
	 */
	Function<Double, Double> logistic = x -> 1.0 / (1 + Math.exp(-x));

	/**
	 * Compute the gradient of loss function w.r.t w
	 * 
	 * @param student
	 * @param preference
	 * @param colleges
	 * @param w
	 * @return Gradient of the CE loss function over the transformation matrix w
	 */
	double[][] gradient(double[] student, int[] preference, double[][] colleges, double[][] w) {
		int k = preference.length;
		List<Integer> prefList = Arrays.stream(preference).boxed().collect(Collectors.toList());

		int count = 0;
		int ds = w.length, dc = w[0].length;
		double[][] g = new double[ds][dc];
		for (int i = 0; i < k; i++) {
			int c1 = preference[i];
			for (int c2 = 0; c2 < colleges.length; c2++) {
				int idx = prefList.indexOf(c2);
				double dr = 0;
				if (idx == -1)
					dr = k - i;
				else if (idx > i)
					dr = idx - i;
				else
					continue;

				count++;
				double[] dif = MathLib.Matrix.subtract(colleges[c1], colleges[c2]);
				double p = logistic.apply(MathLib.Matrix.dotProd(student, MathLib.Matrix.ax(w, dif)));
				double phat = logistic.apply(dr);
				g = MathLib.Matrix.add(g, MathLib.Matrix.multiply(MathLib.Matrix.outerProd(student, dif), p - phat));
			}
		}
		return MathLib.Matrix.multiply(g, 1.0 / count);
	}

	/**
	 * Compute the similarities for a student to all known colleges
	 * 
	 * @param student
	 * @param colleges
	 * @param w
	 * @return Similarities of a given student to colleges
	 */
	double[] sim(double[] student, double[][] colleges, double[][] w) {
		int nc = colleges.length;
		double[] pred = new double[nc];
		for (int c = 0; c < nc; c++) {
			pred[c] = MathLib.Matrix.dotProd(student, MathLib.Matrix.ax(w, colleges[c]));
		}
		return pred;
	}

	/**
	 * Making top-k predictions for one student
	 * 
	 * @param student
	 * @param colleges
	 * @param w
	 * @param k
	 *            top-k
	 * @return Predicted preference ranking
	 */
	public int[] predict(double[] student, double[][] colleges, double[][] w, int k) {
		double[] pred = sim(student, colleges, w);
		int[] rank = MathLib.getRank(pred, false);
		return Arrays.copyOf(rank, k);
	}

	/**
	 * Making predictions on a student's preference
	 * 
	 * @param student
	 * @param colleges
	 * @param w
	 * @return Full preference ranking over colleges
	 */
	public int[] predict(double[] student, double[][] colleges, double[][] w) {
		double[] pred = sim(student, colleges, w);
		int[] rank = MathLib.getRank(pred, false);
		return rank;
	}

	/**
	 * Online enhancing the current model based on stochastic gradient descent
	 * method with additional information of some other students' preferences
	 * 
	 * @param w
	 * @param colleges
	 *            complete list of colleges
	 * @param students
	 *            selected or newly joined students
	 * @param preferences
	 *            ranking preference of the mentioned students over colleges
	 */
	public double[][] enhance(double[][] w, double[][] colleges, double[][] students, List<int[]> preferences) {
		int ns = students.length;
		for (int iter = 0; iter < maxIter; iter++) {
			int i = MathLib.Rand.sample(0, ns);
			double[][] g = gradient(students[i], preferences.get(i), colleges, w);
			w = MathLib.Matrix.subtract(w, MathLib.Matrix.multiply(g, -eta));
		}
		return w;
	}

	/**
	 * Simulation
	 */
	public void simulate() {
		int ns = 100, ds = 25;
		int nc = 200, dc = 23;
		int maxK = 10;

		double[][] students = DataUtil.generateFeatureMatrix(ns, ds, -1, 1);
		double[][] colleges = DataUtil.generateFeatureMatrix(nc, dc, 0, 1);

		List<int[]> preferences = DataUtil.generatePreferences(students, colleges, maxK);
		double[][] w = learn(colleges, students, preferences);

		DataUtil.write(students, "./students.txt");
		DataUtil.write(colleges, "./colleges.txt");
		DataUtil.writePreferences(preferences, "./preferences.txt");
		DataUtil.writeModel(w, "./model.txt");
	}

	public static void main(String[] args) {
		if (args == null) {
			System.err.println("USAGE:\n -train <colleges> <students> <preferences> <model>");
			System.err.println(" -predict <model> <colleges> <student> <k> <preference>");
			System.err.println(" -enhance <model> <colleges> <students> <preferences> <enhence>");
			System.err.println(" -simulate");
			return;
		}

		CollegeMatch cm = new CollegeMatch();
		int m = args.length;
		String prefix = args[0];
		if (prefix.startsWith("-train")) {
			if (m < 5) {
				System.err.println("USAGE: -train <colleges> <students> <preferences> <model>");
				return;
			}

			double[][] colleges = DataUtil.loadTrainingData(args[1]);
			double[][] students = DataUtil.loadTrainingData(args[2]);
			List<int[]> preferences = DataUtil.loadPreferences(args[3]);
			double[][] w = cm.learn(colleges, students, preferences);
			DataUtil.writeModel(w, args[4]);
			return;
		}

		if (prefix.startsWith("-predict")) {
			if (m < 6) {
				System.err.println("USAGE: -predict <model> <colleges> <student> <k> <preference>");
				return;
			}

			double[][] w = DataUtil.loadModel(args[1]);
			double[][] colleges = DataUtil.loadTrainingData(args[2]);
			double[] student = DataUtil.loadInstance(args[3]);

			int k = Integer.parseInt(args[4]);
			int[] preference = null;
			if (k <= 0)
				preference = cm.predict(student, colleges, w);
			else
				preference = cm.predict(student, colleges, w, k);
			DataUtil.writePreference(preference, args[5]);
			return;
		}

		if (prefix.startsWith("-enhance")) {
			if (m < 6) {
				System.err.println("USAGE: -enhance <model> <colleges> <students> <preferences> <enhence>");
				return;
			}

			double[][] w = DataUtil.loadModel(args[1]);
			double[][] colleges = DataUtil.loadTrainingData(args[2]);
			double[][] students = DataUtil.loadTrainingData(args[3]);
			List<int[]> preferences = DataUtil.loadPreferences(args[4]);
			w = cm.enhance(w, colleges, students, preferences);
			DataUtil.writeModel(w, args[5]);
			return;
		}

		if (prefix.startsWith("-simulate")) {
			cm.simulate();
			return;
		}

		System.err.println("ERROR: Undefined operation \"" + prefix + "\"");
		System.err.println("USAGE:\n -train <colleges> <students> <preferences> <model>");
		System.err.println(" -predict <model> <colleges> <student> <k> <preference>");
		System.err.println(" -enhance <model> <colleges> <students> <preferences> <enhence>");
		System.err.println(" -simulate");
		return;
	}
}