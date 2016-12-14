package edu.rpi;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A Mathematical Utility.
 * 
 * @author Chunheng Jiang
 * @version 3.0
 * @since 20131208
 * @since 20161112
 */
public final class MathLib {
	public static class Norm {
		/**
		 * Euclidean Norm
		 * 
		 * @param x
		 * @return sqrt(∑x[i]^2)
		 */
		public static <T extends Number> double euclidean(T[] x) {
			double sum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				sum += x[i].doubleValue() * x[i].doubleValue();
			return Math.sqrt(sum);
		}

		public static double euclidean(double[] x) {
			double sum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				sum += x[i] * x[i];
			return Math.sqrt(sum);
		}

		public static float euclidean(float[] x) {
			float sum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				sum += x[i] * x[i];
			return (float) Math.sqrt(sum);
		}

		public static double euclidean(int[] x) {
			double sum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				sum += x[i] * x[i];
			return Math.sqrt(sum);
		}

		public static <T extends Number> double euclidean(List<T> x) {
			double sum = 0;
			int n = x.size();
			for (int i = 0; i < n; i++) {
				double e = x.get(i).doubleValue();
				sum += e * e;
			}
			return Math.sqrt(sum);
		}

		/**
		 * Weighted Euclidean Norm
		 * 
		 * @param x
		 * @param w
		 *            weight vector
		 * @return sqrt(∑w[i]*x[i]^2)
		 */
		public static <T extends Number, P extends Number> double euclidean(T[] x, P[] w) {
			double wsum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				wsum += w[i].doubleValue() * x[i].doubleValue() * x[i].doubleValue();
			return Math.sqrt(wsum);
		}

		public static double euclidean(double[] x, double[] w) {
			double wsum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				wsum += w[i] * x[i] * x[i];

			return Math.sqrt(wsum);
		}

		public static float euclidean(float[] x, float[] w) {
			float wsum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				wsum += w[i] * x[i] * x[i];

			return (float) Math.sqrt(wsum);
		}

		public static double euclidean(List<? extends Number> x, List<? extends Number> w) {
			double wsum = 0;
			int len = x.size();
			for (int i = 0; i < len; i++) {
				double val = x.get(i).doubleValue();
				wsum += w.get(i).doubleValue() * val * val;
			}
			return Math.sqrt(wsum);
		}

		/**
		 * L_2 Norm, aka Euclidean Norm
		 * 
		 * @param x
		 * @return ∑x[i]^2
		 */
		public static <T extends Number> double l2(T[] x) {
			return euclidean(x);
		}

		public static double l2(double[] x) {
			return euclidean(x);
		}

		public static float l2(float[] x) {
			return euclidean(x);
		}

		public static double l2(int[] x) {
			return euclidean(x);
		}

		public static double l2(List<? extends Number> x) {
			return euclidean(x);
		}

		/**
		 * L_1 Norm
		 * 
		 * @param x
		 * @return ∑|x[i]|
		 */
		public static <T extends Number> double l1(T[] x) {
			double sum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				sum += Math.abs(x[i].doubleValue());
			return sum;
		}

		public static double l1(double[] x) {
			double sum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				sum += Math.abs(x[i]);
			return sum;
		}

		public static float l1(float[] x) {
			float sum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				sum += Math.abs(x[i]);
			return sum;
		}

		public static int l1(int[] x) {
			int sum = 0;
			int len = x.length;
			for (int i = 0; i < len; i++)
				sum += Math.abs(x[i]);
			return sum;
		}

		public static <T extends Number> double l1(List<T> x) {
			double sum = 0;
			int len = x.size();
			for (int i = 0; i < len; i++)
				sum += Math.abs(x.get(i).doubleValue());
			return sum;
		}
	}

	public static class Scale {
		public static <T extends Number> Double[] sum(T[] x) {
			double sum = Data.sum(x);
			Double[] ret = null;
			if (sum != 0) {
				int len = x.length;
				ret = new Double[len];
				for (int i = 0; i < len; i++)
					ret[i] = x[i].doubleValue() / sum;
			}
			return ret;
		}

		public static void sum(double[] array) {
			double sum = Data.sum(array);
			if (sum != 0) {
				int len = array.length;
				for (int i = 0; i < len; i++)
					array[i] /= sum;
			}
		}

		public static void sum(float[] array) {
			float sum = Data.sum(array);
			if (sum != 0) {
				int len = array.length;
				for (int i = 0; i < len; i++)
					array[i] /= sum;
			}
		}

		public static double[] sum(int[] array) {
			double norm = Data.sum(array);
			double[] ret = null;
			if (norm != 0) {
				int len = array.length;
				ret = new double[len];
				for (int i = 0; i < len; i++)
					ret[i] /= norm;
			}
			return ret;
		}

		public static List<Double> sum(List<? extends Number> list) {
			double norm = Data.sum(list);
			List<Double> ret = null;
			if (norm != 0) {
				ret = new ArrayList<>();
				int len = list.size();
				for (int i = 0; i < len; i++)
					ret.add(list.get(i).doubleValue() / norm);
			}
			return ret;
		}

		/**
		 * Normalize Array Based on Maximum Value
		 * 
		 * @param array
		 */
		public static <T extends Number> Double[] max(T[] array) {
			int len = 0;
			if (array == null || (len = array.length) == 0) {
				System.err.println("Array is Empty.");
				System.exit(0);
			}

			Double[] ret = new Double[len];
			double max = array[0].doubleValue();
			if (array.length == 1)
				ret[0] = max;

			for (int i = 1; i < len; i++)
				max = max > array[i].doubleValue() ? max : array[i].doubleValue();

			if (max != 0) {
				for (int i = 0; i < len; i++)
					ret[i] = array[i].doubleValue() / max;
			}
			return ret;
		}

		public static void max(double[] array) {
			double max = Data.max(array);
			if (max != 0) {
				int len = array.length;
				for (int i = 0; i < len; i++)
					array[i] /= max;
			}
		}

		public static void max(float[] array) {
			float max = Data.max(array);
			if (max != 0) {
				int len = array.length;
				for (int i = 0; i < len; i++)
					array[i] /= max;
			}
		}

		public static double[] max(int[] array) {
			double norm = Data.max(array);
			double[] ret = null;
			if (norm != 0) {
				int len = array.length;
				ret = new double[len];
				for (int i = 0; i < len; i++)
					ret[i] /= norm;
			}
			return ret;
		}

		public static List<Double> max(List<? extends Number> list) {
			int len = 0;
			if (list == null || (len = list.size()) == 0) {
				System.err.println("Array is Empty.");
				System.exit(0);
			}

			List<Double> ret = new ArrayList<>();
			double max = list.get(0).doubleValue();
			if (len == 1)
				ret.add(max);

			for (int i = 1; i < len; i++) {
				max = max > list.get(i).doubleValue() ? max : list.get(i).doubleValue();
				ret.add(0.0);
			}

			if (max != 0) {
				for (int i = 0; i < len; i++)
					ret.set(i, list.get(i).doubleValue() / max);
			}
			return ret;
		}

		/**
		 * ZScore Normalization
		 * 
		 * @param array
		 */
		public static <T extends Number> Double[] zscore(T[] array) {
			int n = array.length;
			double mean = Data.sum(array) / n;
			double std = 0;
			for (T t : array) {
				double val = t.doubleValue() - mean;
				std += val * val;
			}
			Double[] ret = new Double[n];
			if (std == 0) {
				mean = 0;
				std = 1;
			} else
				std = Math.sqrt(std / (n - 1));

			for (int i = 0; i < n; i++)
				ret[i] = (array[i].doubleValue() - mean) / std;
			return ret;
		}

		public static void zscore(double[] array) {
			int n = array.length;
			double mean = Data.sum(array) / n;
			double std = 0;
			for (int i = 0; i < n; i++) {
				double val = array[i] - mean;
				std += val * val;
			}
			if (std > 0) {
				std = Math.sqrt(std / (n - 1));
				for (int i = 0; i < n; i++)
					array[i] = (array[i] - mean) / std;
			}
		}

		public static void zscore(float[] array) {
			int n = array.length;
			float mean = Data.sum(array) / n;
			float std = 0;
			for (int i = 0; i < n; i++) {
				float val = array[i] - mean;
				std += val * val;
			}
			if (std > 0) {
				std = (float) Math.sqrt(std / (n - 1));
				for (int i = 0; i < n; i++)
					array[i] = (array[i] - mean) / std;
			}
		}

		public static double[] zscore(int[] array) {
			int len = array.length;
			double mean = Data.mean(array);
			double std = 0;
			for (int i = 0; i < len; i++) {
				double val = array[i] - mean;
				std += val * val;
			}

			double[] ret = new double[len];
			if (std > 0) {
				std = Math.sqrt(std / (len - 1));
				for (int i = 0; i < len; i++)
					ret[i] = (array[i] - mean) / std;
			}
			return ret;
		}

		public static List<Double> zscore(List<? extends Number> x) {
			int n = x.size();
			double mean = Data.mean(x);
			double std = 0;
			for (int i = 0; i < n; i++) {
				double val = x.get(i).doubleValue() - mean;
				std += val * val;
			}
			if (std == 0) {
				mean = 0;
				std = 1;
			} else
				std = Math.sqrt(std / (n - 1));
			List<Double> ret = new ArrayList<>();
			for (int i = 0; i < n; i++)
				ret.add((x.get(i).doubleValue() - mean) / std);
			return ret;
		}

		/**
		 * Normalizing to [a, b]
		 * 
		 * @param x
		 */
		public static <T extends Number> Double[] scale(T[] x, double a, double b) {
			double max = Double.MIN_VALUE;
			double min = Double.MAX_VALUE;

			int n = x.length;
			double val = 0;
			for (T t : x) {
				val = t.doubleValue();
				if (val > max)
					max = val;
				if (val < min)
					min = val;
			}

			Double[] ret = new Double[n];
			double alpha = -1;
			if (max > min & b > a) {
				alpha = (b - a) / (max - min);
				for (int i = 0; i < n; i++)
					ret[i] = alpha * (x[i].doubleValue() - min) + a;
			} else
				for (int i = 0; i < n; i++)
					ret[i] = x[i].doubleValue();
			return ret;
		}

		public static void scale(double[] x, double a, double b) {
			double max = Data.max(x);
			double min = Data.min(x);

			int sz = x.length;
			double alpha = -1;
			if (max > min) {
				alpha = (b - a) / (max - min);
				for (int i = 0; i < sz; i++)
					x[i] = alpha * (x[i] - min) + a;
			}
		}

		public static void scale(float[] x, double a, double b) {
			double max = Data.max(x);
			double min = Data.min(x);

			int n = x.length;
			double alpha = -1;
			if (max > min) {
				alpha = (b - a) / (max - min);
				for (int i = 0; i < n; i++)
					x[i] = (float) (alpha * (x[i] - min) + a);
			}
		}

		public static double[] scale(int[] x, double a, double b) {
			double max = Data.max(x);
			double min = Data.min(x);

			int n = x.length;
			double[] ret = new double[n];
			double alpha = -1;
			if (max > min) {
				alpha = (b - a) / (max - min);
				for (int i = 0; i < n; i++)
					ret[i] = alpha * (x[i] - min) + a;
			} else
				for (int i = 0; i < n; i++)
					ret[i] = x[i] * 1.0;
			return ret;
		}

		public static List<Double> scale(List<? extends Number> x, double a, double b) {
			double max = Double.MIN_VALUE;
			double min = Double.MAX_VALUE;

			int n = x.size();
			double val = 0;
			for (int i = 0; i < n; i++) {
				val = x.get(i).doubleValue();
				if (val > max)
					max = val;
				if (val < min)
					min = val;
			}

			List<Double> ret = new ArrayList<>();
			double alpha = -1;
			if (max > min & b > a) {
				alpha = (b - a) / (max - min);
				for (int i = 0; i < n; i++)
					ret.add(alpha * (x.get(i).doubleValue() - min) + a);
			} else
				for (int i = 0; i < n; i++)
					ret.add(x.get(i).doubleValue());
			return ret;
		}
	}

	public static class Data {
		/**
		 * @param x
		 * @return ∑x[i]
		 */
		public static <T extends Number> double sum(T[] x) {
			double sum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				sum += x[i].doubleValue();
			return sum;
		}

		public static double sum(double[] x) {
			double sum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				sum += x[i];
			return sum;
		}

		public static float sum(float[] x) {
			float sum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				sum += x[i];
			return sum;
		}

		public static int sum(int[] x) {
			int sum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				sum += x[i];
			return sum;
		}

		public static double sum(Collection<? extends Number> x) {
			double sum = 0;
			Iterator<? extends Number> iter = x.iterator();
			while (iter.hasNext())
				sum += iter.next().doubleValue();
			return sum;
		}

		/**
		 * @param x
		 * @return mean
		 */
		public static <T extends Number> double mean(T[] x) {
			int n = x.length;
			return Data.sum(x) * 1.0 / n;
		}

		public static double mean(double[] x) {
			int n = x.length;
			return Data.sum(x) / n;
		}

		public static float mean(float[] x) {
			int n = x.length;
			return Data.sum(x) / n;
		}

		public static double mean(int[] x) {
			int n = x.length;
			return Data.sum(x) * 1.0 / n;
		}

		public static double mean(Collection<? extends Number> x) {
			int n = x.size();
			return Data.sum(x) * 1.0 / n;
		}

		public static double[] median(double[] x) {
			int n = x.length;
			double[] array = new double[n];
			for (int i = 0; i < n; i++)
				array[i] = x[i];
			Arrays.sort(array);
			int mid = n / 2;

			if (n % 2 == 0) {
				double left = array[mid - 1];
				double right = array[mid];
				if (left == right)
					return new double[] { left };
				else
					return new double[] { left, right };
			} else
				return new double[] { array[mid] };
		}

		public static float[] median(float[] x) {
			int n = x.length;
			float[] array = new float[n];
			for (int i = 0; i < n; i++)
				array[i] = x[i];
			Arrays.sort(array);

			int mid = n / 2;

			if (n % 2 == 0) {
				float left = array[mid - 1];
				float right = array[mid];
				if (left == right)
					return new float[] { left };
				else
					return new float[] { left, right };
			} else
				return new float[] { array[mid] };
		}

		public static int[] median(int[] x) {
			int n = x.length;
			int[] array = new int[n];
			for (int i = 0; i < n; i++)
				array[i] = x[i];
			Arrays.sort(array);

			int mid = n / 2;

			if (n % 2 == 0) {
				int left = array[mid - 1];
				int right = array[mid];
				if (left == right)
					return new int[] { left };
				else
					return new int[] { left, right };
			} else
				return new int[] { array[mid] };
		}

		public static <T extends Comparable<? super T>> List<T> median(T[] x) {
			List<T> list = Arrays.stream(x).sorted().collect(Collectors.toList());
			int n = list.size();
			int mid = n / 2;

			if (n % 2 == 0) {
				T left = list.get(mid - 1);
				T right = list.get(mid);
				if (left.equals(right))
					return Arrays.asList(left);
				else
					return Arrays.asList(left, right);
			} else
				return Arrays.asList(list.get(mid));
		}

		public static <T extends Comparable<? super T>> List<T> median(Collection<T> x) {
			List<T> list = new ArrayList<>(x);
			Collections.sort(list);
			int n = list.size();
			int mid = n / 2;

			if (n % 2 == 0) {
				T left = list.get(mid - 1);
				T right = list.get(mid);
				if (left.equals(right))
					return Arrays.asList(left);
				else
					return Arrays.asList(left, right);
			} else
				return Arrays.asList(list.get(mid));
		}

		public static <T extends Comparable<? super T>> List<T> median1(Collection<T> x) {
			Queue<T> minHeap = new PriorityQueue<>();
			Queue<T> maxHeap = new PriorityQueue<>((e1, e2) -> e2.compareTo(e1));

			int index = 0;
			for (T elem : x) {
				maxHeap.add(elem);
				if (index % 2 == 0) {
					if (minHeap.isEmpty()) {
						index++;
						continue;
					} else if (maxHeap.peek().compareTo(minHeap.peek()) > 0) {
						T maxHeapRoot = maxHeap.poll();
						T minHeapRoot = minHeap.poll();
						maxHeap.add(minHeapRoot);
						minHeap.add(maxHeapRoot);
					}
				} else
					minHeap.add(maxHeap.poll());
				index++;
			}

			if (index % 2 == 0) {
				T maxHeapRoot = maxHeap.peek();
				T minHeapRoot = minHeap.peek();
				if (maxHeapRoot.equals(minHeapRoot))
					return Arrays.asList(minHeapRoot);
				else
					return Arrays.asList(maxHeap.peek(), minHeap.peek());
			} else
				return Arrays.asList(maxHeap.peek());
		}

		public static <T extends Number> double variance(T[] x) {
			int n = x.length;
			if (n == 1)
				return 0;

			double mean = mean(x), sum = 0;
			for (T val : x)
				sum += (val.doubleValue() - mean) * (val.doubleValue() - mean);
			return sum / (n - 1);
		}

		public static double variance(double[] x) {
			int n = x.length;
			if (n == 1)
				return 0;

			double mean = mean(x), sum = 0;
			for (double val : x)
				sum += (val - mean) * (val - mean);

			return sum / (n - 1);
		}

		public static float variance(float[] x) {
			int n = x.length;
			if (n == 1)
				return 0;

			float mean = mean(x), sum = 0;
			for (float val : x)
				sum += (val - mean) * (val - mean);
			return sum / (n - 1);
		}

		public static double variance(int[] x) {
			int n = x.length;
			if (n == 1)
				return 0;
			double mean = mean(x), sum = 0;
			for (int val : x)
				sum += (val - mean) * (val - mean);
			return sum / (n - 1);
		}

		public static <T extends Number> double variance(List<T> x) {
			int n = x.size();
			if (n == 1)
				return 0;

			double mean = mean(x);
			double sum = 0;
			for (T t : x)
				sum += (t.doubleValue() - mean) * (t.doubleValue() - mean);
			return sum / (n - 1);
		}

		public static double stdVariance(double[] x) {
			return Math.sqrt(variance(x));
		}

		public static double stdVariance(float[] x) {
			return Math.sqrt(variance(x));
		}

		public static double stdVariance(int[] x) {
			return Math.sqrt(variance(x));
		}

		public static <T extends Number> double stdVariance(List<T> x) {
			return Math.sqrt(variance(x));
		}

		/**
		 * @param x
		 * @return sum in rows
		 */
		public static double[] sumR(double[][] x) {
			int m = x.length;
			double[] sum = new double[m];
			for (int i = 0; i < m; i++)
				sum[i] = sum(x[i]);
			return sum;
		}

		public static float[] sumR(float[][] x) {
			int m = x.length;
			float[] sum = new float[m];
			for (int i = 0; i < m; i++)
				sum[i] = sum(x[i]);
			return sum;
		}

		public static int[] sumR(int[][] x) {
			int m = x.length;
			int[] sum = new int[m];
			for (int i = 0; i < m; i++)
				sum[i] = sum(x[i]);
			return sum;
		}

		/**
		 * @param x
		 * @return sum in columns
		 */
		public static double[] sumC(double[][] x) {
			int n = x[0].length;
			double[] sum = new double[n];
			// column
			for (int j = 0; j < n; j++)
				// row
				for (int i = 0; i < n; i++)
					sum[j] += x[i][j];
			return sum;
		}

		public static float[] sumC(float[][] x) {
			int n = x[0].length;
			float[] sum = new float[n];
			// column
			for (int j = 0; j < n; j++)
				// row
				for (int i = 0; i < n; i++)
					sum[j] += x[i][j];
			return sum;
		}

		public static int[] sumC(int[][] x) {
			int n = x[0].length;
			int[] sum = new int[n];
			// column
			for (int j = 0; j < n; j++)
				// row
				for (int i = 0; i < n; i++)
					sum[j] += x[i][j];
			return sum;
		}

		public static double[] meanR(double[][] x) {
			double[] sumR = sumR(x);
			return Matrix.multiply(sumR, 1.0 / x.length);
		}

		public static float[] meanR(float[][] x) {
			float[] sumR = sumR(x);
			return Matrix.multiply(sumR, 1.0F / x.length);
		}

		public static double[] meanR(int[][] x) {
			int[] sumR = sumR(x);
			return Matrix.multiply(sumR, 1.0 / x.length);
		}

		public static double[] meanC(double[][] x) {
			double[] sumC = sumC(x);
			return Matrix.multiply(sumC, 1.0 / x[0].length);
		}

		public static float[] meanC(float[][] x) {
			float[] sumC = sumC(x);
			return Matrix.multiply(sumC, 1.0F / x[0].length);
		}

		public static double[] meanC(int[][] x) {
			int[] sumC = sumC(x);
			return Matrix.multiply(sumC, 1.0 / x[0].length);
		}

		public static double[][] covariance(double[][] x) {
			double[] meanC = meanC(x);
			int m = x.length;
			int n = x[0].length;

			double[][] sigma = new double[n][n];
			for (int i = 0; i < m; i++) {
				sigma = Matrix.add(sigma, Matrix.aaT(Matrix.subtract(x[i], meanC)));
			}
			return sigma;
		}

		public static float[][] covariance(float[][] x) {
			float[] meanC = meanC(x);
			int m = x.length;
			int n = x[0].length;

			float[][] sigma = new float[n][n];
			for (int i = 0; i < m; i++) {
				sigma = Matrix.add(sigma, Matrix.aaT(Matrix.subtract(x[i], meanC)));
			}
			return sigma;
		}

		public static double[][] covariance(int[][] x) {
			double[] meanC = meanC(x);
			int m = x.length;
			int n = x[0].length;

			double[][] sigma = new double[n][n];
			for (int i = 0; i < m; i++) {
				double[] y = Matrix.multiply(x[i], 1.0);
				sigma = Matrix.add(sigma, Matrix.aaT(Matrix.subtract(y, meanC)));
			}
			return sigma;
		}

		/**
		 * @param x
		 * @return largest value
		 */
		public static <T extends Comparable<? super T>> T max(T[] x) {
			if (x == null) {
				System.err.println("ERROR: Array is Empty.");
				System.exit(0);
			}

			T max = x[0];
			if (x.length == 1)
				return max;

			int n = x.length;
			for (int i = 1; i < n; i++)
				max = max.compareTo(x[i]) > 0 ? max : x[i];

			return max;
		}

		public static double max(double[] x) {
			if (x == null) {
				System.err.println("Array is Empty.");
				System.exit(0);
			}

			double max = x[0];
			if (x.length == 1)
				return max;

			int n = x.length;
			for (int i = 1; i < n; i++)
				max = (max > x[i]) ? max : x[i];

			return max;
		}

		public static float max(float[] x) {
			if (x == null) {
				System.err.println("Array is Empty.");
				System.exit(0);
			}

			float max = x[0];

			if (x.length == 1)
				return max;

			int n = x.length;
			for (int i = 1; i < n; i++)
				max = (max > x[i]) ? max : x[i];

			return max;
		}

		public static int max(int[] x) {
			if (x == null) {
				System.err.println("Array is Empty.");
				System.exit(0);
			}

			int max = x[0];

			if (x.length == 1)
				return max;

			int n = x.length;
			for (int i = 1; i < n; i++)
				max = (max > x[i]) ? max : x[i];

			return max;
		}

		public static <T extends Number> T max(List<T> x) {
			if (x.isEmpty()) {
				System.err.println("List is Empty.");
				System.exit(0);
			}

			T max = x.get(0);

			int n = x.size();
			if (n == 1)
				return max;

			for (int i = 1; i < n; i++)
				max = (max.doubleValue() > x.get(i).doubleValue()) ? max : x.get(i);

			return max;
		}

		/**
		 * Search the smallest element
		 * 
		 * @param x
		 * @return smallest
		 */
		public static <T extends Number> T min(T[] x) {
			if (x == null) {
				System.err.println("Array is Empty.");
				System.exit(0);
			}

			T min = x[0];
			if (x.length == 1)
				return min;

			int n = x.length;
			for (int i = 1; i < n; i++)
				min = (min.doubleValue() < x[i].doubleValue()) ? min : x[i];
			return min;
		}

		public static double min(double[] x) {
			if (x == null) {
				System.err.println("Array is Empty.");
				System.exit(0);
			}

			double min = x[0];
			if (x.length == 1)
				return min;

			int n = x.length;
			for (int i = 1; i < n; i++)
				min = (min < x[i]) ? min : x[i];

			return min;
		}

		public static float min(float[] x) {
			if (x == null) {
				System.err.println("Array is Empty.");
				System.exit(0);
			}

			float min = x[0];
			if (x.length == 1)
				return min;

			int n = x.length;
			for (int i = 1; i < n; i++)
				min = (min < x[i]) ? min : x[i];

			return min;
		}

		public static int min(int[] x) {
			if (x == null) {
				System.err.println("Array is Empty.");
				System.exit(0);
			}

			int min = x[0];

			int n = x.length;
			if (n == 1)
				return min;

			for (int i = 1; i < n; i++)
				min = (min < x[i]) ? min : x[i];

			return min;
		}

		public static <T extends Number> T min(List<T> x) {
			if (x.isEmpty()) {
				System.err.println("List is Empty.");
				System.exit(0);
			}

			T min = x.get(0);

			int n = x.size();
			if (n == 1)
				return min;

			for (int i = 1; i < n; i++)
				min = (min.doubleValue() < x.get(i).doubleValue()) ? min : x.get(i);

			return min;
		}

		public static List<Integer> distinct(int[] x) {
			return Arrays.stream(x).boxed().distinct().collect(Collectors.toList());
		}

		public static List<Integer> distinct(List<Integer> x) {
			return x.stream().distinct().collect(Collectors.toList());
		}

		/**
		 * @param x
		 * @return count each unique value
		 */
		public static <T> Map<T, Long> count(Collection<T> x) {
			Map<T, Long> countTable = new HashMap<>();
			for (T elem : x) {
				long count = 1;
				if (countTable.containsKey(elem))
					count = countTable.get(elem) + 1;
				countTable.put(elem, count);
			}
			return countTable;
		}

		public static <T> Map<T, Long> count(T[] x) {
			Map<T, Long> countTable = new HashMap<>();
			for (T elem : x) {
				long count = 1;
				if (countTable.containsKey(elem))
					count = countTable.get(elem) + 1;
				countTable.put(elem, count);
			}
			return countTable;
		}

		public static Map<Integer, Long> count(int[] x) {
			Map<Integer, Long> countTable = new HashMap<>();
			for (int elem : x) {
				long count = 1;
				if (countTable.containsKey(elem))
					count = countTable.get(elem) + 1;
				countTable.put(elem, count);
			}
			return countTable;
		}

		public static Map<Double, Long> count(double[] x) {
			Map<Double, Long> countTable = new HashMap<>();
			for (double elem : x) {
				long count = 1;
				if (countTable.containsKey(elem))
					count = countTable.get(elem) + 1;
				countTable.put(elem, count);
			}
			return countTable;
		}

		public static Map<Float, Long> count(float[] x) {
			Map<Float, Long> countTable = new HashMap<>();
			for (float elem : x) {
				long count = 1;
				if (countTable.containsKey(elem))
					count = countTable.get(elem) + 1;
				countTable.put(elem, count);
			}
			return countTable;
		}

		public static <T> Map<T, Float> freq(Collection<T> x) {
			Map<T, Long> countTable = count(x);
			int n = x.size();
			Map<T, Float> freq = new HashMap<>();
			for (T key : countTable.keySet())
				freq.put(key, countTable.get(key) * 1.0F / n);
			return freq;
		}

		public static <T> Map<T, Float> freq(T[] x) {
			Map<T, Long> countTable = count(x);
			int n = x.length;
			Map<T, Float> freq = new HashMap<>();
			for (T key : countTable.keySet())
				freq.put(key, countTable.get(key) * 1.0F / n);
			return freq;
		}

		public static Map<Integer, Float> freq(int[] x) {
			Map<Integer, Long> countTable = count(x);
			int n = x.length;
			Map<Integer, Float> freq = new HashMap<>();
			for (int key : countTable.keySet())
				freq.put(key, countTable.get(key) * 1.0F / n);
			return freq;
		}

		public static Map<Float, Float> freq(float[] x) {
			Map<Float, Long> countTable = count(x);
			int n = x.length;
			Map<Float, Float> freq = new HashMap<>();
			for (float key : countTable.keySet())
				freq.put(key, countTable.get(key) * 1.0F / n);
			return freq;
		}

		public static Map<Double, Float> freq(double[] x) {
			Map<Double, Long> countTable = count(x);
			int n = x.length;
			Map<Double, Float> freq = new HashMap<>();
			for (double key : countTable.keySet())
				freq.put(key, countTable.get(key) * 1.0F / n);
			return freq;
		}

		public static <T> List<T> mode(List<T> x) {
			Map<T, Long> countTable = count(x);
			List<T> modes = new ArrayList<>();
			long max = 0;
			for (Map.Entry<T, Long> entry : countTable.entrySet()) {
				long count = entry.getValue();
				if (count < max)
					continue;

				if (count > max) {
					max = count;
					modes.clear();
				}
				modes.add(entry.getKey());
			}
			return modes;
		}

		public static int[] mode(int[] x) {
			Map<Integer, Long> countTable = count(x);
			List<Integer> modeList = new ArrayList<>();
			long max = 0;
			for (Map.Entry<Integer, Long> entry : countTable.entrySet()) {
				long count = entry.getValue();
				if (count < max)
					continue;

				if (count > max) {
					max = count;
					modeList.clear();
				}
				modeList.add(entry.getKey());
			}

			int[] modes = new int[modeList.size()];
			int i = 0;
			for (int mode : modeList)
				modes[i++] = mode;
			return modes;
		}

		public static double[] mode(double[] x) {
			Map<Double, Long> countTable = count(x);
			List<Double> modeList = new ArrayList<>();
			long max = 0;
			for (Map.Entry<Double, Long> entry : countTable.entrySet()) {
				long count = entry.getValue();
				if (count < max)
					continue;

				if (count > max) {
					max = count;
					modeList.clear();
				}
				modeList.add(entry.getKey());
			}

			double[] modes = new double[modeList.size()];
			int i = 0;
			for (double mode : modeList)
				modes[i++] = mode;
			return modes;
		}

		public static float[] mode(float[] x) {
			Map<Float, Long> countTable = count(x);
			List<Float> modeList = new ArrayList<>();
			long max = 0;
			for (Map.Entry<Float, Long> entry : countTable.entrySet()) {
				long count = entry.getValue();
				if (count < max)
					continue;

				if (count > max) {
					max = count;
					modeList.clear();
				}
				modeList.add(entry.getKey());
			}

			float[] modes = new float[modeList.size()];
			int i = 0;
			for (float mode : modeList)
				modes[i++] = mode;
			return modes;
		}

		static final long[] FACTORIALS = new long[] { 1l, 1l, 2l, 6l, 24l, 120l, 720l, 5040l, 40320l, 362880l, 3628800l,
				39916800l, 479001600l, 6227020800l, 87178291200l, 1307674368000l, 20922789888000l, 355687428096000l,
				6402373705728000l, 121645100408832000l, 2432902008176640000l };

		public static long factorial(int n) {
			if (n < 0) {
				System.err.println("ERROR: No definition of factorial for negative integers.");
				return -1;
			} else if (n > 20) {
				System.err.println("ERROR: The factorial value is too large to fit in a long.");
				return -1;
			} else
				return FACTORIALS[n];
		}
	}

	/**
	 * Basic matrix operation, including matrix addition, subtraction,
	 * multiplication, scalar addition and multiplication, transpose, norm and
	 * trace operation on a vector.
	 */
	public static class Matrix {

		/**
		 * @param x
		 * @param y
		 * @return z[i] = x[i] + y[i]
		 */
		public static <T extends Number, P extends Number> Double[] add(T[] x, P[] y) {
			int n = x.length;
			Double[] z = new Double[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i].doubleValue() + y[i].doubleValue();
			return z;
		}

		public static double[] add(double[] x, double[] y) {
			int n = x.length;
			double[] z = new double[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i] + y[i];
			return z;
		}

		public static float[] add(float[] x, float[] y) {
			int n = x.length;
			float[] z = new float[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i] + y[i];
			return z;
		}

		public static int[] add(int[] x, int[] y) {
			int n = x.length;
			int[] z = new int[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i] + y[i];
			return z;
		}

		public static <T extends Number> List<Double> add(List<T> x, List<T> y) {
			int n = x.size();
			List<Double> z = new ArrayList<>();
			for (int i = 0; i < n; i++)
				z.add(x.get(i).doubleValue() + y.get(i).doubleValue());
			return z;
		}

		public static int[][] add(int[][] x, int[][] y) {
			int m = x.length;
			int n = x[0].length;
			if (m != y.length || n != y[0].length)
				throw new IllegalArgumentException("Inconsistent Dimension");
			int[][] z = new int[m][n];
			for (int i = 0; i < m; i++)
				z[i] = add(x[i], y[i]);
			return z;
		}

		public static float[][] add(float[][] x, float[][] y) {
			int m = x.length;
			int n = x[0].length;
			if (m != y.length || n != y[0].length)
				throw new IllegalArgumentException("Inconsistent Dimension");
			float[][] z = new float[m][n];
			for (int i = 0; i < m; i++)
				z[i] = add(x[i], y[i]);
			return z;
		}

		public static double[][] add(double[][] x, double[][] y) {
			int m = x.length;
			int n = x[0].length;
			if (m != y.length || n != y[0].length)
				throw new IllegalArgumentException("Inconsistent Dimension");

			double[][] z = new double[m][n];
			for (int i = 0; i < m; i++)
				z[i] = add(x[i], y[i]);
			return z;
		}

		/**
		 * Subtraction
		 * 
		 * @param x
		 * @param y
		 * @return z[i] = x[i] - y[i]
		 */
		public static <T extends Number, P extends Number> Double[] subtract(T[] x, P[] y) {
			int n = x.length;
			Double[] z = new Double[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i].doubleValue() - y[i].doubleValue();
			return z;
		}

		public static double[] subtract(double[] x, double[] y) {
			int n = x.length;
			double[] z = new double[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i] - y[i];
			return z;
		}

		public static float[] subtract(float[] x, float[] y) {
			int n = x.length;
			float[] z = new float[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i] - y[i];
			return z;
		}

		public static int[] subtract(int[] x, int[] y) {
			int n = x.length;
			int[] z = new int[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i] - y[i];
			return z;
		}

		public static <T extends Number, P extends Number> List<Double> subtract(List<T> x, List<P> y) {
			int n = x.size();
			List<Double> z = new ArrayList<>();
			for (int i = 0; i < n; i++)
				z.add(x.get(i).doubleValue() - y.get(i).doubleValue());
			return z;
		}

		public static double[][] subtract(double[][] x, double[][] y) {
			int m = x.length, n = x[0].length;
			if (m != y.length || n != y[0].length)
				throw new IllegalArgumentException("Inconsistent Dimension");

			double[][] z = new double[m][n];
			for (int i = 0; i < m; i++)
				z[i] = add(x[i], y[i]);
			return z;
		}

		public static float[][] subtract(float[][] x, float[][] y) {
			int m = x.length, n = x[0].length;
			if (m != y.length || n != y[0].length)
				throw new IllegalArgumentException("Inconsistent Dimension");

			float[][] z = new float[m][n];
			for (int i = 0; i < m; i++)
				z[i] = subtract(x[i], y[i]);
			return z;
		}

		public static int[][] subtract(int[][] x, int[][] y) {
			int m = x.length, n = x[0].length;
			if (m != y.length || n != y[0].length)
				throw new IllegalArgumentException("Inconsistent Dimension");

			int[][] z = new int[m][n];
			for (int i = 0; i < m; i++)
				z[i] = subtract(x[i], y[i]);
			return z;
		}

		public static <T extends Number> Double[] multiply(T[] x, double a) {
			int n = x.length;
			Double[] y = new Double[n];
			for (int i = 0; i < n; i++)
				y[i] = x[i].doubleValue() * a;
			return y;
		}

		public static double[] multiply(double[] x, double a) {
			int n = x.length;
			double[] y = new double[n];
			for (int i = 0; i < n; i++)
				y[i] = x[i] * a;
			return y;
		}

		public static float[] multiply(float[] x, float a) {
			int n = x.length;
			float[] y = new float[n];
			for (int i = 0; i < n; i++)
				y[i] = x[i] * a;

			return y;
		}

		public static int[] multiply(int[] x, int a) {
			int n = x.length;
			int[] arr = new int[n];
			for (int i = 0; i < n; i++)
				arr[i] = x[i] * a;
			return arr;
		}

		public static double[] multiply(int[] x, double a) {
			int n = x.length;
			double[] y = new double[n];
			for (int i = 0; i < n; i++)
				y[i] = x[i] * a;
			return y;
		}

		public static List<Double> multiply(List<? extends Number> x, double a) {
			int n = x.size();
			List<Double> y = new ArrayList<>();
			for (int i = 0; i < n; i++)
				y.add(x.get(i).doubleValue() * a);
			return y;
		}

		public static double[][] multiply(double[][] x, double c) {
			int n = x.length;
			double[][] y = new double[n][];
			for (int i = 0; i < n; i++)
				y[i] = multiply(x[i], c);
			return y;
		}

		/**
		 * Update an array by adding a multiple of another array y = a * x + y.
		 */
		public static void axPy(double a, double[] x, double[] y) {
			if (x.length != y.length) {
				throw new IllegalArgumentException(
						String.format("Arrays have different length: x[%d], y[%d]", x.length, y.length));
			}

			for (int i = 0; i < x.length; i++) {
				y[i] += a * x[i];
			}
		}

		/**
		 * Product of a matrix and a vector y = A * x according to the rules of
		 * linear algebra. Number of columns in A must equal number of elements
		 * in x.
		 */
		public static void ax(double[][] a, double[] x, double[] y) {
			if (a[0].length != x.length) {
				throw new IllegalArgumentException(
						String.format("Array dimensions do not match for matrix multiplication: %dx%d vs %dx1",
								a.length, a[0].length, x.length));
			}

			if (a.length != y.length) {
				throw new IllegalArgumentException(String.format("Array dimensions do not match"));
			}

			Arrays.fill(y, 0.0);
			for (int i = 0; i < y.length; i++) {
				for (int k = 0; k < a[i].length; k++) {
					y[i] += a[i][k] * x[k];
				}
			}
		}

		public static double[] ax(double[][] a, double[] x) {
			int n = a[0].length;
			if (n != x.length) {
				throw new IllegalArgumentException(
						String.format("Array dimensions do not match for matrix multiplication: %dx%d vs %dx1",
								a.length, a[0].length, x.length));
			}

			int m = a.length;
			double[] y = new double[m];
			for (int i = 0; i < m; i++) {
				for (int k = 0; k < n; k++) {
					y[i] += a[i][k] * x[k];
				}
			}
			return y;
		}

		/**
		 * Product of a matrix and a vector y = A * x + y according to the rules
		 * of linear algebra. Number of columns in A must equal number of
		 * elements in x.
		 */
		public static void axPy(double[][] a, double[] x, double[] y) {
			if (a[0].length != x.length) {
				throw new IllegalArgumentException(
						String.format("Array dimensions do not match for matrix multiplication: %dx%d vs %dx1",
								a.length, a[0].length, x.length));
			}

			if (a.length != y.length) {
				throw new IllegalArgumentException(String.format("Array dimensions do not match"));
			}

			for (int i = 0; i < y.length; i++) {
				for (int k = 0; k < a[i].length; k++) {
					y[i] += a[i][k] * x[k];
				}
			}
		}

		/**
		 * Product of a matrix and a vector y = A * x + b * y according to the
		 * rules of linear algebra. Number of columns in A must equal number of
		 * elements in x.
		 */
		public static void axPby(double[][] a, double[] x, double[] y, double b) {
			if (a[0].length != x.length) {
				throw new IllegalArgumentException(
						String.format("Array dimensions do not match for matrix multiplication: %dx%d vs %dx1",
								a.length, a[0].length, x.length));
			}

			if (a.length != y.length) {
				throw new IllegalArgumentException(String.format("Array dimensions do not match"));
			}

			for (int i = 0; i < y.length; i++) {
				y[i] *= b;
				for (int k = 0; k < a[i].length; k++) {
					y[i] += a[i][k] * x[k];
				}
			}
		}

		/**
		 * Product of a matrix and a vector y = A<sup>T</sup> * x according to
		 * the rules of linear algebra. Number of elements in x must equal
		 * number of rows in A.
		 */
		public static void aTx(double[][] a, double[] x, double[] y) {
			if (a.length != x.length) {
				throw new IllegalArgumentException(
						String.format("Array dimensions do not match for matrix multiplication: %d x %d vs 1 x %d",
								a.length, a[0].length, x.length));
			}

			if (a[0].length != y.length) {
				throw new IllegalArgumentException(String.format("Array dimensions do not match"));
			}

			Arrays.fill(y, 0.0);
			for (int i = 0; i < y.length; i++) {
				for (int k = 0; k < x.length; k++) {
					y[i] += x[k] * a[k][i];
				}
			}
		}

		/**
		 * Product of a matrix and a vector y = A<sup>T</sup> * x + y according
		 * to the rules of linear algebra. Number of elements in x must equal
		 * number of rows in A.
		 */
		public static void aTxPy(double[][] a, double[] x, double[] y) {
			if (a.length != x.length) {
				throw new IllegalArgumentException(
						String.format("Array dimensions do not match for matrix multiplication: 1 x %d vs %d x %d",
								x.length, a.length, a[0].length));
			}

			if (a[0].length != y.length) {
				throw new IllegalArgumentException(String.format("Array dimensions do not match"));
			}

			for (int i = 0; i < y.length; i++) {
				for (int k = 0; k < x.length; k++) {
					y[i] += x[k] * a[k][i];
				}
			}
		}

		/**
		 * Product of a matrix and a vector y = A<sup>T</sup> * x + b * y
		 * according to the rules of linear algebra. Number of elements in x
		 * must equal number of rows in A.
		 */
		public static void aTxPby(double[][] a, double[] x, double[] y, double b) {
			if (a.length != x.length) {
				throw new IllegalArgumentException(
						String.format("Array dimensions do not match for matrix multiplication: 1 x %d vs %d x %d",
								x.length, a.length, a[0].length));
			}

			if (a[0].length != y.length) {
				throw new IllegalArgumentException(String.format("Array dimensions do not match"));
			}

			for (int i = 0; i < y.length; i++) {
				y[i] *= b;
				for (int k = 0; k < x.length; k++) {
					y[i] += x[k] * a[k][i];
				}
			}
		}

		/**
		 * Returns x' * A * x.
		 */
		public static double xAx(double[][] a, double[] x) {
			if (a.length != a[0].length) {
				throw new IllegalArgumentException("The matrix is not square");
			}

			if (a.length != x.length) {
				throw new IllegalArgumentException(
						String.format("x' * A * x: 1 x %d vs %d x %d", x.length, a.length, a[0].length));
			}

			int n = a.length;
			double s = 0.0;
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					s += a[i][j] * x[i] * x[j];
				}
			}

			return s;
		}

		/**
		 * 
		 * @param a
		 *            vector
		 * @return a*a'
		 */
		public static double[][] aaT(double[] a) {
			int m = a.length;
			double[][] c = new double[m][m];
			for (int i = 0; i < m; i++)
				for (int j = 0; j < m; j++)
					c[i][j] = a[i] * a[j];
			return c;
		}

		public static float[][] aaT(float[] a) {
			int m = a.length;
			float[][] c = new float[m][m];
			for (int i = 0; i < m; i++)
				for (int j = 0; j < m; j++)
					c[i][j] = a[i] * a[j];
			return c;
		}

		public static int[][] aaT(int[] a) {
			int m = a.length;
			int[][] c = new int[m][m];
			for (int i = 0; i < m; i++)
				for (int j = 0; j < m; j++)
					c[i][j] = a[i] * a[j];
			return c;
		}

		/**
		 * 
		 * @param a
		 * @param c
		 *            = a * a'
		 */
		public static void aaT(double[] a, double[][] c) {
			int m = a.length;
			for (int i = 0; i < m; i++)
				for (int j = 0; j < m; j++)
					c[i][j] = a[i] * a[j];
		}

		public static void aaT(float[] a, float[][] c) {
			int m = a.length;
			for (int i = 0; i < m; i++)
				for (int j = 0; j < m; j++)
					c[i][j] = a[i] * a[j];
		}

		public static void aaT(int[] a, int[][] c) {
			int m = a.length;
			for (int i = 0; i < m; i++)
				for (int j = 0; j < m; j++)
					c[i][j] = a[i] * a[j];
		}

		/**
		 * Matrix multiplication A * A' according to the rules of linear
		 * algebra.
		 */
		public static double[][] aaT(double[][] a) {
			int m = a.length;
			int n = a[0].length;
			double[][] c = new double[m][m];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < m; j++) {
					for (int k = 0; k < n; k++) {
						c[i][j] += a[i][k] * a[j][k];
					}
				}
			}

			return c;
		}

		/**
		 * Matrix multiplication C = A * A' according to the rules of linear
		 * algebra.
		 */
		public static void aaT(double[][] a, double[][] c) {
			int m = a.length;
			int n = a[0].length;

			for (int i = 0; i < m; i++) {
				for (int j = 0; j < m; j++) {
					for (int k = 0; k < n; k++) {
						c[i][j] += a[i][k] * a[j][k];
					}
				}
			}
		}

		/**
		 * Matrix multiplication A' * A according to the rules of linear
		 * algebra.
		 */
		public static double[][] aTa(double[][] a) {
			int m = a.length;
			int n = a[0].length;

			double[][] c = new double[n][n];
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < m; k++) {
						c[i][j] += a[k][i] * a[k][j];
					}
				}
			}
			return c;
		}

		/**
		 * Matrix multiplication C = A' * A according to the rules of linear
		 * algebra.
		 */
		public static void aTa(double[][] a, double[][] c) {
			int m = a.length;
			int n = a[0].length;

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < m; k++) {
						c[i][j] += a[k][i] * a[k][j];
					}
				}
			}
		}

		/**
		 * Matrix multiplication A * B according to the rules of linear algebra.
		 */
		public static double[][] ab(double[][] a, double[][] b) {
			if (a[0].length != b.length) {
				throw new IllegalArgumentException(String.format("Matrix multiplication A * B: %d x %d vs %d x %d",
						a.length, a[0].length, b.length, b[0].length));
			}

			int m = a.length;
			int n = b[0].length;
			int l = b.length;
			double[][] c = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < l; k++) {
						c[i][j] += a[i][k] * b[k][j];
					}
				}
			}

			return c;
		}

		/**
		 * Matrix multiplication C = A * B according to the rules of linear
		 * algebra.
		 */
		public static void ab(double[][] a, double[][] b, double[][] c) {
			if (a[0].length != b.length) {
				throw new IllegalArgumentException(String.format("Matrix multiplication A * B: %d x %d vs %d x %d",
						a.length, a[0].length, b.length, b[0].length));
			}

			int m = a.length;
			int n = b[0].length;
			int l = b.length;

			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < l; k++) {
						c[i][j] += a[i][k] * b[k][j];
					}
				}
			}
		}

		/**
		 * Matrix multiplication A' * B according to the rules of linear
		 * algebra.
		 */
		public static double[][] aTb(double[][] a, double[][] b) {
			if (a.length != b.length) {
				throw new IllegalArgumentException(String.format("Matrix multiplication A' * B: %d x %d vs %d x %d",
						a.length, a[0].length, b.length, b[0].length));
			}

			int m = a[0].length;
			int n = b[0].length;
			int l = b.length;
			double[][] c = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < l; k++) {
						c[i][j] += a[k][i] * b[k][j];
					}
				}
			}

			return c;
		}

		/**
		 * Matrix multiplication C = A' * B according to the rules of linear
		 * algebra.
		 */
		public static void aTb(double[][] a, double[][] b, double[][] c) {
			if (a.length != b.length) {
				throw new IllegalArgumentException(String.format("Matrix multiplication A' * B: %d x %d vs %d x %d",
						a.length, a[0].length, b.length, b[0].length));
			}

			int m = a[0].length;
			int n = b[0].length;
			int l = b.length;

			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < l; k++) {
						c[i][j] += a[k][i] * b[k][j];
					}
				}
			}
		}

		/**
		 * Matrix multiplication A * B' according to the rules of linear
		 * algebra.
		 */
		public static double[][] abT(double[][] a, double[][] b) {
			if (a[0].length != b[0].length) {
				throw new IllegalArgumentException(String.format("Matrix multiplication A * B': %d x %d vs %d x %d",
						a.length, a[0].length, b.length, b[0].length));
			}

			int m = a.length;
			int n = b.length;
			int l = b[0].length;
			double[][] c = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < l; k++) {
						c[i][j] += a[i][k] * b[j][k];
					}
				}
			}

			return c;
		}

		/**
		 * Matrix multiplication C = A * B' according to the rules of linear
		 * algebra.
		 */
		public static void abT(double[][] a, double[][] b, double[][] c) {
			if (a[0].length != b[0].length) {
				throw new IllegalArgumentException(String.format("Matrix multiplication A * B': %d x %d vs %d x %d",
						a.length, a[0].length, b.length, b[0].length));
			}

			int m = a.length;
			int n = b.length;
			int l = b[0].length;

			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < l; k++) {
						c[i][j] += a[i][k] * b[j][k];
					}
				}
			}
		}

		/**
		 * Matrix multiplication A' * B' according to the rules of linear
		 * algebra.
		 */
		public static double[][] aTbT(double[][] a, double[][] b) {
			if (a.length != b[0].length) {
				throw new IllegalArgumentException(String.format("Matrix multiplication A' * B': %d x %d vs %d x %d",
						a.length, a[0].length, b.length, b[0].length));
			}

			int m = a[0].length;
			int n = b.length;
			int l = a.length;
			double[][] c = new double[m][n];
			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < l; k++) {
						c[i][j] += a[k][i] * b[j][k];
					}
				}
			}
			return c;
		}

		/**
		 * Matrix multiplication C = A' * B' according to the rules of linear
		 * algebra.
		 */
		public static void aTbT(double[][] a, double[][] b, double[][] c) {
			if (a.length != b[0].length) {
				throw new IllegalArgumentException(String.format("Matrix multiplication A' * B': %d x %d vs %d x %d",
						a.length, a[0].length, b.length, b[0].length));
			}

			int m = a[0].length;
			int n = b.length;
			int l = a.length;

			for (int i = 0; i < m; i++) {
				for (int j = 0; j < n; j++) {
					for (int k = 0; k < l; k++) {
						c[i][j] += a[k][i] * b[j][k];
					}
				}
			}
		}

		public static double trace(double[][] x) {
			int n = Math.min(x.length, x[0].length);
			double ret = 0.0;
			for (int i = 0; i < n; i++)
				ret += x[i][i];
			return ret;
		}

		public static float trace(float[][] x) {
			int n = Math.min(x.length, x[0].length);
			float ret = 0;
			for (int i = 0; i < n; i++)
				ret += x[i][i];
			return ret;
		}

		public static double[][] transpose(double[][] x) {
			int m = x.length;
			int n = x[0].length;

			double[][] xT = new double[n][m];
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					xT[j][i] = x[i][j];
			return xT;
		}

		public static float[][] transpose(float[][] x) {
			int m = x.length;
			int n = x[0].length;

			float[][] xT = new float[n][m];
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					xT[j][i] = x[i][j];
			return xT;
		}

		/**
		 * Linear Combination
		 * 
		 * @param x
		 * @param w1
		 * @param y
		 * @param w2
		 * @return z[i] = x[i]*w1 + y[i]*w2
		 */
		public static <T extends Number, P extends Number> Double[] lin(T[] x, double w1, P[] y, double w2) {
			if (x == null || y == null) {
				System.err.println("ERROR: Empty Array.");
				System.exit(0);
			}
			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			int n = x.length;
			Double[] z = new Double[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i].doubleValue() * w1 + y[i].doubleValue() * w2;
			return z;
		}

		public static double[] lin(double[] x, double w1, double[] y, double w2) {
			if (x == null || y == null) {
				System.err.println("ERROR: Empty Array.");
				System.exit(0);
			}
			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			int n = x.length;
			double[] z = new double[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i] * w1 + y[i] * w2;
			return z;
		}

		public static float[] lin(float[] x, float w1, float[] y, float w2) {
			if (x == null || y == null) {
				System.err.println("ERROR: Empty Array.");
				System.exit(0);
			}
			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			int n = x.length;
			float[] z = new float[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i] * w1 + y[i] * w2;
			return z;
		}

		public static double[] lin(int[] x, double w1, int[] y, double w2) {
			if (x == null || y == null) {
				System.err.println("ERROR: Empty Array.");
				System.exit(0);
			}
			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			int n = x.length;
			double[] z = new double[n];
			for (int i = 0; i < n; i++)
				z[i] = x[i] * w1 + y[i] * w2;
			return z;
		}

		public static <T extends Number, P extends Number> List<Double> lin(List<T> x, P w1, List<T> y, P w2) {
			if (x.isEmpty() || y.isEmpty()) {
				System.err.println("ERROR: Empty Array.");
				System.exit(0);
			}
			if (x.size() != y.size()) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			int n = x.size();
			List<Double> z = new ArrayList<>();
			for (int i = 0; i < n; i++)
				z.add(x.get(i).doubleValue() * w1.doubleValue() + y.get(i).doubleValue() * w2.doubleValue());
			return z;
		}

		/**
		 * Inner Product
		 * 
		 * @param x
		 * @param y
		 * @return ∑arr1[i]*arr2[i]
		 */
		public static <T extends Number, P extends Number> double innerProd(T[] x, P[] y) {
			if (x == null || y == null) {
				System.err.println("ERROR: Empty Array.");
				System.exit(0);
			}

			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			double sum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				sum += x[i].doubleValue() * y[i].doubleValue();
			return sum;
		}

		public static double innerProd(double[] x, double[] y) {
			if (x == null || y == null) {
				System.err.println("ERROR: Empty Array.");
				System.exit(0);
			}
			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			double sum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				sum += x[i] * y[i];

			return sum;
		}

		public static float innerProd(float[] x, float[] y) {
			if (x == null || y == null) {
				System.err.println("Empty Array.");
				System.exit(0);
			}
			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			float sum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				sum += x[i] * y[i];

			return sum;
		}

		public static int innerProd(int[] x, int[] y) {
			if (x == null || y == null) {
				System.err.println("Empty Array.");
				System.exit(0);
			}
			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			int sum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				sum += x[i] * y[i];
			return sum;
		}

		public static double innerProd(List<? extends Number> x, List<? extends Number> y) {
			if (x.isEmpty() || y.isEmpty()) {
				System.err.println("Empty List.");
				System.exit(0);
			}
			if (x.size() != y.size()) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}
			double sum = 0;
			int n = x.size();
			for (int i = 0; i < n; i++)
				sum += x.get(i).doubleValue() * y.get(i).doubleValue();
			return sum;
		}

		/**
		 * Weigthed Inner Product
		 * 
		 * @param x
		 * @param y
		 * @return ∑w[i]arr1[i]*arr2[i]
		 */
		public static <T extends Number, P extends Number> double innerProd(T[] x, T[] y, P[] w) {
			if (x == null || y == null) {
				System.err.println("Empty Array.");
				System.exit(0);
			}

			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			double wsum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				wsum += w[i].doubleValue() * x[i].doubleValue() * y[i].doubleValue();

			return wsum;
		}

		public static double innerProd(double[] x, double[] y, double[] w) {
			if (x == null || y == null) {
				System.err.println("Empty Array.");
				System.exit(0);
			}

			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			double wsum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				wsum += w[i] * x[i] * y[i];

			return wsum;
		}

		public static float innerProd(float[] x, float[] y, float[] w) {
			if (x == null || y == null) {
				System.err.println("Empty Array.");
				System.exit(0);
			}

			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			float wsum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				wsum += w[i] * x[i] * y[i];

			return wsum;
		}

		public static double innerProd(int[] x, int[] y, double[] w) {
			if (x == null || y == null) {
				System.err.println("Empty Array.");
				System.exit(0);
			}

			if (x.length != y.length) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			double wsum = 0;
			int n = x.length;
			for (int i = 0; i < n; i++)
				wsum += w[i] * x[i] * y[i];

			return wsum;
		}

		public static double innerProd(List<? extends Number> x, List<? extends Number> y, List<? extends Number> w) {
			if (x.isEmpty() || y.isEmpty()) {
				System.err.println("Empty List.");
				System.exit(0);
			}
			if (x.size() != y.size()) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			double wsum = 0;
			int n = x.size();
			for (int i = 0; i < n; i++)
				wsum += w.get(i).doubleValue() * x.get(i).doubleValue() * y.get(i).doubleValue();

			return wsum;
		}

		/**
		 * Dot Product, alias of inner product
		 * 
		 * @param x
		 * @param y
		 * @return ∑x[i]*y[i]
		 */
		public static <T extends Number, P extends Number> double dotProd(T[] x, P[] y) {
			return innerProd(x, y);
		}

		public static double dotProd(double[] x, double[] y) {
			return innerProd(x, y);
		}

		public static float dotProd(float[] x, float[] y) {
			return innerProd(x, y);
		}

		public static int dotProd(int[] x, int[] y) {
			return innerProd(x, y);
		}

		public static double dotProd(List<? extends Number> x, List<? extends Number> y) {
			return innerProd(x, y);
		}

		public static double[][] outerProd(double[] x, double[] y) {
			int m = x.length;
			int n = y.length;
			double[][] r = new double[m][n];
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					r[i][j] = x[i] * y[j];
			return r;
		}

		public static float[][] outerProd(float[] x, float[] y) {
			int m = x.length;
			int n = y.length;
			float[][] r = new float[m][n];
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					r[i][j] = x[i] * y[j];
			return r;
		}

		public static int[][] outerProd(int[] x, int[] y) {
			int m = x.length;
			int n = y.length;
			int[][] r = new int[m][n];
			for (int i = 0; i < m; i++)
				for (int j = 0; j < n; j++)
					r[i][j] = x[i] * y[j];
			return r;
		}

		public static <T extends Number, P extends Number> double affline(T[] x, P[] y, boolean tailBias) {
			if (x == null || y == null) {
				System.err.println("ERROR: Empty array.");
				System.exit(0);
			}

			if (x.length != y.length + 1) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			double sum = 0.0;
			if (tailBias) {
				for (int i = 0; i < y.length; i++)
					sum += x[i].doubleValue() * y[i].doubleValue();
				return sum + x[x.length - 1].doubleValue();
			}
			for (int i = 1; i < x.length; i++)
				sum += x[i].doubleValue() * y[i - 1].doubleValue();
			return sum + x[0].doubleValue();
		}

		public static double affline(double[] x, double[] y, boolean tailBias) {
			if (x == null || y == null) {
				System.err.println("ERROR: Empty array.");
				System.exit(0);
			}

			if (x.length != y.length + 1) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			double sum = 0.0;
			if (tailBias) {
				for (int i = 0; i < y.length; i++)
					sum += x[i] * y[i];
				return sum + x[x.length - 1];
			}
			for (int i = 1; i < x.length; i++)
				sum += x[i] * y[i - 1];
			return sum + x[0];
		}

		public static float affline(float[] x, float[] y, boolean tailBias) {
			if (x == null || y == null) {
				System.err.println("ERROR: Empty array.");
				System.exit(0);
			}

			if (x.length != y.length + 1) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			float sum = 0.0F;
			if (tailBias) {
				for (int i = 0; i < y.length; i++)
					sum += x[i] * y[i];
				return sum + x[x.length - 1];
			}
			for (int i = 1; i < x.length; i++)
				sum += x[i] * y[i - 1];
			return sum + x[0];
		}

		public static int affline(int[] x, int[] y, boolean tailBias) {
			if (x == null || y == null) {
				System.err.println("ERROR: Empty array.");
				System.exit(0);
			}

			if (x.length != y.length + 1) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			int sum = 0;
			if (tailBias) {
				for (int i = 0; i < y.length; i++)
					sum += x[i] * y[i];
				return sum + x[x.length - 1];
			}
			for (int i = 1; i < x.length; i++)
				sum += x[i] * y[i - 1];
			return sum + x[0];
		}

		/**
		 * @param x
		 *            a list contains a bias
		 * @param y
		 * @param tailBias
		 * @return x^T y + b
		 */
		public static double affline(List<? extends Number> x, List<? extends Number> y, boolean tailBias) {
			if (x.isEmpty() || y.isEmpty()) {
				System.err.println("ERROR: Empty List.");
				System.exit(0);
			}

			if (x.size() != y.size() + 1) {
				System.err.println("ERROR: Dimensions are inconsistent.");
				System.exit(0);
			}

			double sum = 0;
			if (tailBias) {
				for (int i = 0; i < y.size(); i++)
					sum += x.get(i).doubleValue() * y.get(i).doubleValue();
				return sum + x.get(x.size() - 1).doubleValue();
			}
			for (int i = 1; i < x.size(); i++)
				sum += x.get(i).doubleValue() * y.get(i - 1).doubleValue();
			return sum + x.get(0).doubleValue();
		}

		/**
		 * Weigthed Inner Product
		 * 
		 * @param x
		 * @param y
		 * @return ∑w[i]arr1[i]*arr2[i]
		 */
		public static <T extends Number, P extends Number> double dotProd(T[] x, T[] y, P[] w) {
			return innerProd(x, y, w);
		}

		public static double dotProd(double[] x, double[] y, double[] w) {
			return innerProd(x, y, w);
		}

		public static float weightedDotProd(float[] x, float[] y, float[] w) {
			return innerProd(x, y, w);
		}

		public static double dotProd(int[] x, int[] y, double[] w) {
			return innerProd(x, y, w);
		}

		public static double dotProd(List<? extends Number> x, List<? extends Number> y, List<? extends Number> w) {
			return innerProd(x, y, w);
		}

		/**
		 * Normalize Array Based on Euclidean Norm or L_2 Norm
		 * 
		 * @param x
		 */
		public static <T extends Number> Double[] normalize(T[] x) {
			double norm = Norm.l2(x);
			Double[] ret = null;
			if (norm != 0) {
				int n = x.length;
				ret = new Double[n];
				for (int i = 0; i < n; i++)
					ret[i] = x[i].doubleValue() / norm;
			}
			return ret;
		}

		public static void normalize(double[] x) {
			double norm = Norm.l2(x);
			if (norm != 0) {
				int n = x.length;
				for (int i = 0; i < n; i++)
					x[i] /= norm;
			}
		}

		public static void normalize(float[] x) {
			double norm = Norm.l2(x);
			if (norm != 0) {
				int n = x.length;
				for (int i = 0; i < n; i++)
					x[i] /= norm;
			}
		}

		public static double[] normalize(int[] x) {
			double norm = Norm.l2(x);
			double[] ret = null;
			if (norm != 0) {
				int n = x.length;
				ret = new double[n];
				for (int i = 0; i < n; i++)
					ret[i] /= norm;
			}
			return ret;
		}

		public static <T extends Number> List<Double> normalize(List<T> x) {
			double norm = Norm.l2(x);
			List<Double> ret = null;
			if (norm != 0) {
				ret = new ArrayList<Double>();
				int n = x.size();
				for (int i = 0; i < n; i++)
					ret.add(x.get(i).doubleValue() / norm);
			}
			return ret;
		}

		public static void eye(int[][] x) {
			int d = Math.min(x.length, x[0].length);
			for (int i = 0; i < d; i++)
				x[i][i] = 1;
		}

		public static void eye(float[][] x) {
			int d = Math.min(x.length, x[0].length);
			for (int i = 0; i < d; i++)
				x[i][i] = 1;
		}

		public static void eye(double[][] x) {
			int d = Math.min(x.length, x[0].length);
			for (int i = 0; i < d; i++)
				x[i][i] = 1;
		}

		public static void eye(int n) {
			double[][] x = new double[n][n];
			for (int i = 0; i < n; i++)
				x[i][i] = 1;
		}

		public static void ones(int[] x) {
			for (int i = 0; i < x.length; i++)
				x[i] = 1;
		}

		public static void ones(float[] x) {
			for (int i = 0; i < x.length; i++)
				x[i] = 1;
		}

		public static void ones(double[] x) {
			for (int i = 0; i < x.length; i++)
				x[i] = 1;
		}

		public static double[] ones(int n) {
			double[] x = new double[n];
			for (int i = 0; i < n; i++)
				x[i] = 1;
			return x;
		}

		public static void unit(float[] x) {
			int n = x.length;
			for (int i = 0; i < n; i++)
				x[i] = 1.0F / n;
		}

		public static void unit(double[] x) {
			int n = x.length;
			for (int i = 0; i < n; i++)
				x[i] = 1.0 / n;
		}

		public static double[] unit(int n) {
			double[] x = new double[n];
			for (int i = 0; i < n; i++)
				x[i] = 1.0 / n;
			return x;
		}

		/**
		 * @param data
		 *            symmetric matrix
		 * @return ridge-like matrix
		 */
		public static float[][] ridge(float[][] data) {
			List<Integer> idList = new ArrayList<>();
			int dim = data.length;
			for (int i = 1; i < dim; i++)
				idList.add(i);

			int curId = 0;
			List<Integer> ridgeList = new ArrayList<>();
			ridgeList.add(curId);

			getRidgeId(data, ridgeList, idList, curId);

			float[][] ridgeMatrix = new float[dim][dim];
			for (int i = 0; i < dim; i++)
				for (int j = 0; j < dim; j++) {
					if (i > j)
						ridgeMatrix[i][j] = ridgeMatrix[j][i];
					else
						ridgeMatrix[i][j] = data[ridgeList.get(i)][ridgeList.get(j)];
				}
			return ridgeMatrix;
		}

		/**
		 * get diagonal elements' ids in a ridge-like matrix
		 * 
		 * @param matrix
		 *            symmetric matrix
		 * @param ridgeList
		 * @param idxList
		 * @param curID
		 */
		static void getRidgeId(float[][] matrix, List<Integer> ridgeList, List<Integer> idxList, int curID) {
			int sz = 0;
			if (idxList == null || (sz = idxList.size()) == 0)
				return;

			int maxId = idxList.get(0);
			if (sz == 1) {
				ridgeList.add(maxId);
				return;
			}

			float[] row = new float[sz];
			for (int i = 0; i < sz; i++)
				row[i] = matrix[curID][idxList.get(i)];

			int[] rank = getRank(row, false);

			for (int i = 0; i < sz; i++)
				rank[i] = idxList.get(rank[i]);
			idxList = toList(rank);

			maxId = idxList.get(0);
			ridgeList.add(maxId);
			idxList.remove(0);

			getRidgeId(matrix, ridgeList, idxList, maxId);
		}

		/**
		 * @param data
		 *            symmetric matrix
		 * @return ridge form matrix
		 */
		public static double[][] ridge(double[][] data) {
			List<Integer> idList = new ArrayList<>();
			int dim = data.length;
			for (int i = 1; i < dim; i++)
				idList.add(i);

			int curId = 0;
			List<Integer> ridgeList = new ArrayList<>();
			ridgeList.add(curId);

			getRidgeId(data, ridgeList, idList, curId);

			double[][] ridgeMatrix = new double[dim][dim];
			for (int i = 0; i < dim; i++)
				for (int j = 0; j < dim; j++) {
					if (i > j)
						ridgeMatrix[i][j] = ridgeMatrix[j][i];
					else
						ridgeMatrix[i][j] = data[ridgeList.get(i)][ridgeList.get(j)];
				}
			return ridgeMatrix;
		}

		/**
		 * @param matrix
		 *            symmetric matrix
		 * @param ridgeList
		 * @param idxList
		 * @param curID
		 */
		private static void getRidgeId(double[][] matrix, List<Integer> ridgeList, List<Integer> idxList, int curID) {
			int sz = 0;
			if (idxList == null || (sz = idxList.size()) == 0)
				return;

			int maxId = idxList.get(0);
			if (sz == 1) {
				ridgeList.add(maxId);
				return;
			}

			double[] row = new double[sz];
			for (int i = 0; i < sz; i++)
				row[i] = matrix[curID][idxList.get(i)];

			int[] rank = getRank(row, false);

			for (int i = 0; i < sz; i++)
				rank[i] = idxList.get(rank[i]);
			idxList = toList(rank);

			maxId = idxList.get(0);
			ridgeList.add(maxId);
			idxList.remove(0);

			getRidgeId(matrix, ridgeList, idxList, maxId);
		}

		/**
		 * @param data
		 *            symmetric matrix
		 * @return ridge form matrix
		 */
		public static int[][] ridge(int[][] data) {
			List<Integer> idList = new ArrayList<>();
			int dim = data.length;
			for (int i = 1; i < dim; i++)
				idList.add(i);

			int curId = 0;
			List<Integer> ridgeList = new ArrayList<>();
			ridgeList.add(curId);

			getRidgeId(data, ridgeList, idList, curId);

			int[][] ridgeMatrix = new int[dim][dim];
			for (int i = 0; i < dim; i++)
				for (int j = 0; j < dim; j++) {
					if (i > j)
						ridgeMatrix[i][j] = ridgeMatrix[j][i];
					else
						ridgeMatrix[i][j] = data[ridgeList.get(i)][ridgeList.get(j)];
				}
			return ridgeMatrix;
		}

		/**
		 * Obtain the id of the digonal element in a ridge-like matrix
		 * 
		 * @param matrix
		 *            symmetric matrix
		 * @param ridgeList
		 * @param idxList
		 * @param currentId
		 *            focus
		 */
		static void getRidgeId(int[][] matrix, List<Integer> ridgeList, List<Integer> idxList, int currentId) {
			int size = 0;
			if (idxList == null || (size = idxList.size()) == 0)
				return;

			int maxId = idxList.get(0);
			if (size == 1) {
				ridgeList.add(maxId);
				return;
			}

			int[] row = new int[size];
			for (int i = 0; i < size; i++)
				row[i] = matrix[currentId][idxList.get(i)];

			int[] rank = getRank(row, false);

			for (int i = 0; i < size; i++)
				rank[i] = idxList.get(rank[i]);
			idxList = toList(rank);

			maxId = idxList.get(0);
			ridgeList.add(maxId);
			idxList.remove(0);

			getRidgeId(matrix, ridgeList, idxList, maxId);
		}

		/**
		 * @param n
		 * @return stochastic matrix
		 */
		public static double[][] stochastic(int n) {
			double[][] matrix = new double[n][n];
			for (int i = 0; i < n; i++)
				Rand.distribution(matrix[i]);
			return matrix;
		}

		public static void stochastic(float[][] a) {
			int n = a.length;
			for (int i = 0; i < n; i++)
				Rand.distribution(a[i]);
		}

		public static void stochastic(double[][] a) {
			int n = a.length;
			for (int i = 0; i < n; i++)
				Rand.distribution(a[i]);
		}

		/**
		 * Confusion Matrix, primarily used in evaluate classification accuracy.
		 * Each element c(i,j) presents the number of instances in class or
		 * cluster i are predicted with label j
		 * 
		 * @param t
		 *            truth class
		 * @param p
		 *            predicted class
		 */
		public static void confusion(int[] t, int[] p, int[][] c) {
			int n = t.length;
			if (n != p.length)
				throw new IllegalArgumentException("The dimensions are not consistent.");

			for (int i = 0; i < n; i++)
				c[t[i]][p[i]]++;
		}

		/**
		 * Confusion Matrix, primarily used in evaluate classification accuracy.
		 * Each element c(i,j) presents the percentage of instances in class or
		 * cluster i are predicted with label j
		 * 
		 * @param t
		 * @param p
		 * @param c
		 */
		public static void confusion(int[] t, int[] p, float[][] c) {
			int n = t.length;
			if (n != p.length)
				throw new IllegalArgumentException("The dimensions are not consistent.");

			int k = c.length;
			int[] count = new int[k];
			for (int i = 0; i < n; i++) {
				c[t[i]][p[i]]++;
				count[t[i]]++;
			}
			for (int i = 0; i < k; i++)
				Matrix.multiply(c[i], 1.0F / count[i]);
		}

		public static void confusion(int[] t, int[] p, double[][] c) {
			int n = t.length;
			if (n != p.length)
				throw new IllegalArgumentException("The dimensions are not consistent.");

			int k = c.length;
			int[] count = new int[k];
			for (int i = 0; i < n; i++) {
				c[t[i]][p[i]]++;
				count[t[i]]++;
			}
			for (int i = 0; i < k; i++)
				Matrix.multiply(c[i], 1.0 / count[i]);
		}

		public static double[][] distance(int[][] x) {
			int n = x.length;
			double[][] distances = new double[n][n];
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++)
					if (j > i) {
						distances[i][j] = Distance.euclidean(x[i], x[j]);
					} else
						distances[i][j] = distances[j][i];
			}
			return distances;
		}

		public static float[][] distance(float[][] x) {
			int n = x.length;
			float[][] distances = new float[n][n];
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++)
					if (j > i) {
						distances[i][j] = Distance.euclidean(x[i], x[j]);
					} else
						distances[i][j] = distances[j][i];
			}
			return distances;
		}

		public static double[][] distance(double[][] x) {
			int n = x.length;
			double[][] distances = new double[n][n];
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++)
					if (j > i) {
						distances[i][j] = Distance.euclidean(x[i], x[j]);
					} else
						distances[i][j] = distances[j][i];
			}
			return distances;
		}
	}

	public static class Distance {
		/**
		 * Compute Hamming distance of two strings. The two strings in question
		 * must have the same length.
		 * 
		 * @param s1
		 * @param s2
		 * @return number of different characters at the same place
		 */
		public static int hamming(String s1, String s2) {
			int len = s1.length();
			if (len != s2.length()) {
				System.err.println("ERROR: Inconsistent dimensions.");
				System.exit(0);
			}

			int concordant = 0;
			for (int i = 0; i < len; i++) {
				if (s1.charAt(i) == s2.charAt(i))
					concordant++;
			}
			return len - concordant;
		}

		/**
		 * @param x
		 * @param y
		 * @return sqrt(∑(x[i] - y[i]^2))
		 */
		public static <T extends Number, P extends Number> double euclidean(T[] x, P[] y) {
			return Norm.l2(Matrix.subtract(x, y));
		}

		public static double euclidean(double[] x, double[] y) {
			return Norm.l2(Matrix.subtract(x, y));
		}

		public static float euclidean(float[] x, float[] y) {
			return Norm.l2(Matrix.subtract(x, y));
		}

		public static double euclidean(int[] x, int[] y) {
			return Norm.l2(Matrix.subtract(x, y));
		}

		public static double euclidean(List<? extends Number> x, List<? extends Number> y) {
			return Norm.l2(Matrix.subtract(x, y));
		}
	}

	public static class Sim {
		/**
		 * Computing Cosine Similarity between Two Vector
		 * 
		 * @param x
		 * @param y
		 * @return cosine<x, y>
		 */
		public static <T extends Number, P extends Number> double cosine(T[] x, P[] y) {
			if (x == null || y == null) {
				System.err.println("At least one array is Empty.");
				System.exit(0);
			}

			if (x.length != y.length) {
				System.err.println("Dimensions are inconsistent.");
				System.exit(0);
			}

			double norm1 = 0;
			double norm2 = 0;
			double inprod = 0;
			int n = x.length;
			for (int i = 0; i < n; i++) {
				inprod += x[i].doubleValue() * y[i].doubleValue();
				norm1 += x[i].doubleValue() * x[i].doubleValue();
				norm2 += y[i].doubleValue() * y[i].doubleValue();
			}

			double ret = 0;
			if (norm1 * norm2 != 0)
				ret = inprod / (Math.sqrt(norm1 * norm2));
			return ret;
		}

		/**
		 * Computing Cosine Similarity between Two Vector
		 * 
		 * @param x
		 * @param y
		 * @return
		 */
		public static double cosine(double[] x, double[] y) {
			if (x == null || y == null) {
				System.err.println("At least one array is Empty.");
				System.exit(0);
			}

			if (x.length != y.length) {
				System.err.println("Dimensions are inconsistent.");
				System.exit(0);
			}

			double norm1 = 0;
			double norm2 = 0;
			double inprod = 0;
			int n = x.length;
			for (int i = 0; i < n; i++) {
				inprod += x[i] * y[i];
				norm1 += x[i] * x[i];
				norm2 += y[i] * y[i];
			}

			double ret = 0;
			if (norm1 * norm2 != 0)
				ret = inprod / (Math.sqrt(norm1 * norm2));
			return ret;
		}

		public static float cosine(float[] x, float[] y) {
			if (x == null || y == null) {
				System.err.println("At least one array is Empty.");
				System.exit(0);
			}

			if (x.length != y.length) {
				System.err.println("Dimensions are inconsistent.");
				System.exit(0);
			}

			float norm1 = 0;
			float norm2 = 0;
			float inprod = 0;
			int n = x.length;
			for (int i = 0; i < n; i++) {
				inprod += x[i] * y[i];
				norm1 += x[i] * x[i];
				norm2 += y[i] * y[i];
			}

			float ret = 0;
			if (norm1 * norm2 != 0)
				ret = (float) (inprod / (Math.sqrt(norm1 * norm2)));
			return ret;
		}

		public static double cosine(int[] x, int[] y) {
			if (x == null || y == null) {
				System.err.println("At least one array is Empty.");
				System.exit(0);
			}
			if (x.length != y.length) {
				System.err.println("Dimensions are inconsistent.");
				System.exit(0);
			}

			double norm1 = 0;
			double norm2 = 0;
			double inprod = 0;
			int n = x.length;
			for (int i = 0; i < n; i++) {
				inprod += x[i] * y[i];
				norm1 += x[i] * x[i];
				norm2 += y[i] * y[i];
			}

			double ret = 0;
			if (norm1 * norm2 != 0)
				ret = inprod / (Math.sqrt(norm1 * norm2));
			return ret;
		}

		public static <T extends Number, P extends Number> double cosine(List<T> x, List<P> y) {
			if (x.isEmpty() || y.isEmpty()) {
				System.err.println("At least one list is Empty.");
				System.exit(0);
			}

			int n1 = x.size();
			int n2 = y.size();
			if (n1 != n2) {
				System.err.println("Dimensions are inconsistent.");
				System.exit(0);
			}

			double norm1 = 0;
			double norm2 = 0;
			double inprod = 0;
			for (int i = 0; i < n1; i++) {
				double val1 = x.get(i).doubleValue();
				double val2 = y.get(i).doubleValue();
				inprod += val1 * val2;
				norm1 += val1 * val1;
				norm2 += val2 * val2;
			}

			double ret = 0;
			if (norm1 * norm2 != 0)
				ret = inprod / (Math.sqrt(norm1 * norm2));

			return ret;
		}

		/**
		 * Compute the weighted cosine similarity
		 * 
		 * @param x
		 * @param y
		 * @param w
		 * @return weighted cosine similarity
		 */
		public static <T extends Number, P extends Number> double cosine(T[] x, T[] y, P[] w) {
			if (x == null || y == null) {
				System.err.println("At least one array is Empty.");
				System.exit(0);
			}

			double ret = 0;
			if (x.length == y.length && y.length == w.length) {
				double norm1 = 0;
				double norm2 = 0;
				double inprod = 0;
				int n = x.length;
				for (int i = 0; i < n; i++) {
					inprod += x[i].doubleValue() * y[i].doubleValue() * w[i].doubleValue();
					norm1 += x[i].doubleValue() * x[i].doubleValue() * w[i].doubleValue();
					norm2 += y[i].doubleValue() * y[i].doubleValue() * w[i].doubleValue();
				}

				if (norm1 * norm2 != 0)
					ret = inprod / (Math.sqrt(norm1 * norm2));
			} else {
				System.err.println("Dimensions are inconsistent.");
				System.exit(0);
			}

			return ret;
		}

		public static double cosine(double[] x, double[] y, double[] w) {
			if (x == null || y == null) {
				System.err.println("At least one array is Empty.");
				System.exit(0);
			}

			double ret = 0;
			if (x.length == y.length && y.length == w.length) {
				double norm1 = 0;
				double norm2 = 0;
				double inprod = 0;
				int n = x.length;
				for (int i = 0; i < n; i++) {
					inprod += x[i] * y[i] * w[i];
					norm1 += x[i] * x[i] * w[i];
					norm2 += y[i] * y[i] * w[i];
				}

				if (norm1 * norm2 != 0)
					ret = inprod / (Math.sqrt(norm1 * norm2));
			} else {
				System.err.println("Dimensions are inconsistent.");
				System.exit(0);
			}

			return ret;
		}

		public static float cosine(float[] x, float[] y, float[] w) {
			if (x == null || y == null) {
				System.err.println("At least one array is Empty.");
				System.exit(0);
			}

			float ret = 0;
			if (x.length == y.length && y.length == w.length) {
				float norm1 = 0;
				float norm2 = 0;
				float inprod = 0;
				int n = x.length;
				for (int i = 0; i < n; i++) {
					inprod += x[i] * y[i] * w[i];
					norm1 += x[i] * x[i] * w[i];
					norm2 += y[i] * y[i] * w[i];
				}

				if (norm1 * norm2 != 0)
					ret = (float) (inprod / (Math.sqrt(norm1 * norm2)));
			} else {
				System.err.println("Dimensions are inconsistent.");
				System.exit(0);
			}
			return ret;
		}

		public static double cosine(int[] x, int[] y, double[] w) {
			if (x == null || y == null) {
				System.err.println("At least one array is Empty.");
				System.exit(0);
			}

			double ret = 0;
			if (x.length == y.length && y.length == w.length) {
				double norm1 = 0;
				double norm2 = 0;
				double inprod = 0;
				int n = x.length;
				for (int i = 0; i < n; i++) {
					inprod += x[i] * y[i] * w[i];
					norm1 += x[i] * x[i] * w[i];
					norm2 += y[i] * y[i] * w[i];
				}

				if (norm1 * norm2 != 0)
					ret = inprod / (Math.sqrt(norm1 * norm2));
			} else {
				System.err.println("Dimensions are inconsistent.");
				System.exit(0);
			}

			return ret;
		}

		public static <U extends Number, V extends Number> double cosine(List<U> x, List<U> y, List<V> w) {
			if (x.isEmpty() || y.isEmpty()) {
				System.err.println("At least one array is Empty.");
				System.exit(0);
			}

			double ret = 0;
			int n = x.size();
			if (n == y.size() && y.size() == w.size()) {
				double norm1 = 0;
				double norm2 = 0;
				double inprod = 0;
				for (int i = 0; i < n; i++) {
					double val1 = x.get(i).doubleValue();
					double val2 = y.get(i).doubleValue();
					double weight = w.get(i).doubleValue();
					inprod += val1 * val2 * weight;
					norm1 += val1 * val1 * weight;
					norm2 += val2 * val2 * weight;
				}

				if (norm1 * norm2 != 0)
					ret = inprod / (Math.sqrt(norm1 * norm2));

			} else {
				System.err.println("Dimensions are inconsistent.");
				System.exit(0);
			}

			return ret;
		}
	}

	public static class Rand {
		static Random rand = new Random(System.currentTimeMillis());

		/**
		 * Sampling n integers in [a, b) without replacement
		 * 
		 * @param a
		 * @param b
		 * @param n
		 * @return n different integers in [a,b)
		 */
		public static List<Integer> sample(int a, int b, int n) {
			if (b <= a)
				throw new IllegalArgumentException("b must be larger than a");
			else if (b - a < n)
				throw new IllegalArgumentException("[a, b) has less than n integers");

			List<Integer> list = new ArrayList<>();
			for (int i = a; i < b; i++)
				list.add(i);
			Collections.shuffle(list);
			return list.subList(0, n);
		}

		/**
		 * @param a
		 * @param b
		 * @return select one integer in [a, b) randomly
		 */
		public static int sample(int a, int b) {
			if (b <= a)
				throw new IllegalArgumentException("b must be larger than a");
			return a + rand.nextInt(b - a);
		}

		/**
		 * Sampling with replacement
		 * 
		 * @param a
		 * @param b
		 * @param n
		 * @return select n integers in [a, b) randomly
		 */
		public static List<Integer> sampleR(int a, int b, int n) {
			if (b <= a)
				throw new IllegalArgumentException("b must be larger than a");

			int len = b - a;
			List<Integer> list = new ArrayList<>();
			for (int i = 0; i < n; i++)
				list.add(a + rand.nextInt(len));
			return list;
		}

		public static <T> List<T> sampleR(List<T> x, int n) {
			if (x == null || x.size() == 0)
				throw new IllegalArgumentException("empty source");
			int m = x.size();
			int c = 0;
			List<T> ret = new ArrayList<>();
			while (c++ < n)
				ret.add(x.get(rand.nextInt(m)));
			return ret;
		}

		public static <T> T sample(List<T> x) {
			if (x == null || x.size() == 0)
				throw new IllegalArgumentException("empty source");
			return x.get(rand.nextInt(x.size()));
		}

		/**
		 * Produce a number in Gaussian distribution randomly
		 * 
		 * @param mu
		 * @param sigma
		 * @return x~N(mu, sigma)
		 */
		public static double gaussian(double mu, double sigma) {
			return mu + rand.nextGaussian() * sigma;
		}

		public static float gaussian(float mu, float sigma) {
			return (float) (mu + rand.nextGaussian() * sigma);
		}

		/**
		 * Produce uniformly distributed number randomly
		 * 
		 * @param a
		 * @param b
		 * @return x ~ U(a,b)
		 */
		public static double uniform(double a, double b) {
			return a + Math.random() * (b - a);
		}

		public static float uniform(float a, float b) {
			return (float) (a + Math.random() * (b - a));
		}

		public static void distribution(double[] p) {
			int n = p.length;
			double sum = 0;
			for (int i = 0; i < n; i++) {
				p[i] = Math.random();
				sum += p[i];
			}
			p = Matrix.multiply(p, 1.0 / sum);
		}

		public static double[] distribution(int n) {
			double[] p = new double[n];
			double sum = 0;
			for (int i = 0; i < n; i++) {
				p[i] = Math.random();
				sum += p[i];
			}
			return Matrix.multiply(p, 1.0 / sum);
		}

		public static void distribution(float[] p) {
			int n = p.length;
			float sum = 0;
			for (int i = 0; i < n; i++) {
				p[i] = (float) Math.random();
				sum += p[i];
			}
			p = Matrix.multiply(p, 1.0F / sum);
		}

		/**
		 * Randomly generate multinomial samples
		 * 
		 * @param p
		 * @param nTrials
		 * @param nSamples
		 * @return multinomial samples
		 * @see https://en.wikipedia.org/wiki/Multinomial_distribution
		 */
		public static int[][] getMultinomialSamples(double[] p, int nTrials, int nSamples) {
			int range = p.length + 1;
			double[] distribution = new double[range];
			double sum = Data.sum(p);

			distribution[0] = 0;
			for (int i = 1; i < range; i++)
				distribution[i] = distribution[i - 1] + (p[i - 1] / sum);
			distribution[range - 1] = 1.0;

			int[][] sample = new int[nSamples][p.length];
			for (int k = 0; k < nSamples; k++)
				for (int i = 0; i < nTrials; i++)
					sample[k][getMultinomialTrial(distribution, rand.nextDouble())]++;
			return sample;
		}

		/**
		 * generate a multinomial sample randomly
		 * 
		 * @param prob
		 * @param nTrials
		 * @return multinomial sample
		 */
		public static int[] getMultinomialSample(double[] prob, int nTrials) {
			int range = prob.length + 1;
			double[] distribution = new double[range];
			double sum = Data.sum(prob);

			distribution[0] = 0;
			for (int i = 1; i < range; i++)
				distribution[i] = distribution[i - 1] + (prob[i - 1] / sum);

			distribution[range - 1] = 1.0;

			int[] sample = new int[nTrials];
			for (int i = 0; i < nTrials; i++)
				sample[getMultinomialTrial(distribution, rand.nextDouble())]++;
			return sample;
		}

		static int getMultinomialTrial(double[] distribution, double value) {
			int min = 1, max = distribution.length - 1, mid = (min + max) / 2;
			while (min <= max) {
				if (value < distribution[mid - 1]) {
					max = mid - 1;
				} else if (value > distribution[mid]) {
					min = mid + 1;
				} else {
					return mid - 1;
				}
				mid = min + (int) Math.ceil((max - min) / 2);
			}
			return distribution.length - 1;
		}
	}

	/**
	 * @param x
	 * @param ascend
	 * @return rank of array in ascending or descending order
	 */
	public static <T extends Comparable<? super T>> int[] getRank(T[] x, boolean ascend) {
		int n = x.length;
		if (n <= 0)
			return null;

		int inv = ascend ? 1 : -1;
		return IntStream.range(0, n).boxed().sorted((i, j) -> inv * (x[i].compareTo(x[j]))).mapToInt(e -> e).toArray();
	}

	public static int[] getRank(double[] x, boolean ascend) {
		int n = x.length;
		int[] rank = new int[n];
		for (int i = 0; i < n; i++)
			rank[i] = i;

		int idx = (ascend) ? 1 : -1;
		for (int i = 0; i < n - 1; i++) {
			int max = i;
			for (int j = i + 1; j < n; j++)
				if (x[rank[max]] * idx > x[rank[j]] * idx)
					max = j;

			// swap
			int tmp = rank[i];
			rank[i] = rank[max];
			rank[max] = tmp;
		}
		return rank;
	}

	public static int[] getRank(float[] x, boolean ascend) {
		int n = x.length;
		int[] rank = new int[n];
		for (int i = 0; i < n; i++)
			rank[i] = i;

		int idx = (ascend) ? 1 : -1;
		for (int i = 0; i < n - 1; i++) {
			int max = i;
			for (int j = i + 1; j < n; j++)
				if (x[rank[max]] * idx > x[rank[j]] * idx)
					max = j;

			// swap
			int tmp = rank[i];
			rank[i] = rank[max];
			rank[max] = tmp;
		}
		return rank;
	}

	public static int[] getRank(int[] x, boolean ascend) {
		int n = x.length;
		int[] rank = new int[n];
		for (int i = 0; i < n; i++)
			rank[i] = i;

		int idx = (ascend) ? 1 : -1;
		for (int i = 0; i < n - 1; i++) {
			int max = i;
			for (int j = i + 1; j < n; j++)
				if (x[rank[max]] * idx > x[rank[j]] * idx)
					max = j;

			// swap
			int tmp = rank[i];
			rank[i] = rank[max];
			rank[max] = tmp;
		}
		return rank;
	}

	/**
	 * @param x
	 * @param ascend
	 * @return rank of list in ascending or descending order
	 */
	public static <T extends Comparable<? super T>> int[] getRank(List<T> x, boolean ascend) {
		int n = x.size();
		if (n <= 0)
			return null;

		int inv = ascend ? 1 : -1;
		return IntStream.range(0, n).boxed().sorted((i, j) -> inv * (x.get(i).compareTo(x.get(j)))).mapToInt(e -> e)
				.toArray();
	}

	/**
	 * Search all indices of max elements in a list
	 * 
	 * @param x
	 * @return indices of max values
	 */
	public static <T extends Comparable<? super T>> int[] argmax(List<T> x) {
		int n = x.size();
		if (n <= 0)
			return null;

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> x.get(j).compareTo(x.get(i))).mapToInt(e -> e)
				.toArray();

		T max = x.get(rank[0]);
		return Arrays.stream(rank).filter(i -> x.get(i) == max).toArray();
	}

	public static <T, R extends Comparable<? super R>> int[] argmax(List<T> x, Function<T, R> function) {
		int n = x.size();
		if (n <= 0)
			return null;

		List<R> result = x.stream().map(function).collect(Collectors.toList());
		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> result.get(j).compareTo(result.get(i)))
				.mapToInt(e -> e).toArray();

		R max = result.get(rank[0]);
		return Arrays.stream(rank).filter(i -> result.get(i) == max).toArray();
	}

	public static <T extends Comparable<? super T>> int[] argmax(T[] x) {
		int n = x.length;
		if (n <= 0)
			return null;

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> x[j].compareTo(x[i])).mapToInt(e -> e).toArray();

		T max = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == max).toArray();
	}

	public static <T, R extends Comparable<? super R>> int[] argmax(T[] x, Function<T, R> function) {
		int n = x.length;
		if (n <= 0)
			return null;

		List<R> result = Arrays.stream(x).map(function).collect(Collectors.toList());
		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> result.get(j).compareTo(result.get(i)))
				.mapToInt(e -> e).toArray();

		R max = result.get(rank[0]);
		return Arrays.stream(rank).filter(i -> result.get(i) == max).toArray();
	}

	public static int[] argmax(int[] x) {
		int n = x.length;
		if (n <= 0)
			return null;

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> Integer.compare(x[j], x[i])).mapToInt(e -> e)
				.toArray();

		int max = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == max).toArray();
	}

	public static <R extends Comparable<? super R>> int[] argmax(int[] x, Function<Integer, R> function) {
		int n = x.length;
		if (n <= 0)
			return null;

		List<R> result = new ArrayList<>(n);
		for (int elem : x)
			result.add(function.apply(elem));

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> result.get(j).compareTo(result.get(i)))
				.mapToInt(e -> e).toArray();

		int max = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == max).toArray();
	}

	public static int[] argmax(float[] x) {
		int n = x.length;
		if (n <= 0)
			return null;

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> Float.compare(x[j], x[i])).mapToInt(e -> e)
				.toArray();

		float max = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == max).toArray();
	}

	public static <R extends Comparable<? super R>> int[] argmax(float[] x, Function<Float, R> function) {
		int n = x.length;
		if (n <= 0)
			return null;

		List<R> result = new ArrayList<>(n);
		for (float elem : x)
			result.add(function.apply(elem));

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> result.get(j).compareTo(result.get(i)))
				.mapToInt(e -> e).toArray();

		float max = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == max).toArray();
	}

	public static int[] argmax(double[] x) {
		int n = x.length;
		if (n <= 0)
			return null;

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> Double.compare(x[j], x[i])).mapToInt(e -> e)
				.toArray();

		double max = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == max).toArray();
	}

	public static <R extends Comparable<? super R>> int[] argmax(double[] x, Function<Double, R> function) {
		int n = x.length;
		if (n <= 0)
			return null;

		List<R> result = new ArrayList<>(n);
		for (double elem : x)
			result.add(function.apply(elem));

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> result.get(j).compareTo(result.get(i)))
				.mapToInt(e -> e).toArray();

		double max = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == max).toArray();
	}

	/**
	 * Search all indices of min elements in a list
	 * 
	 * @param x
	 * @return indicies of min values
	 */
	public static <T extends Comparable<? super T>> int[] argmin(List<T> x) {
		int n = x.size();
		if (n <= 0)
			return null;

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> x.get(i).compareTo(x.get(j))).mapToInt(e -> e)
				.toArray();

		T min = x.get(rank[0]);
		return Arrays.stream(rank).filter(i -> x.get(i) == min).toArray();
	}

	public static <T, R extends Comparable<? super R>> int[] argmin(List<T> x, Function<T, R> function) {
		int n = x.size();
		if (n <= 0)
			return null;

		List<R> result = x.stream().map(function).collect(Collectors.toList());
		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> result.get(i).compareTo(result.get(j)))
				.mapToInt(e -> e).toArray();

		R min = result.get(rank[0]);
		return Arrays.stream(rank).filter(i -> result.get(i) == min).toArray();
	}

	public static <T extends Comparable<? super T>> int[] argmin(T[] x) {
		int n = x.length;
		if (n <= 0)
			return null;

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> x[i].compareTo(x[j])).mapToInt(e -> e).toArray();

		T min = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == min).toArray();
	}

	public static <T, R extends Comparable<? super R>> int[] argmin(T[] x, Function<T, R> function) {
		int n = x.length;
		if (n <= 0)
			return null;

		List<R> result = Arrays.stream(x).map(function).collect(Collectors.toList());
		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> result.get(i).compareTo(result.get(j)))
				.mapToInt(e -> e).toArray();

		R min = result.get(rank[0]);
		return Arrays.stream(rank).filter(i -> result.get(i) == min).toArray();
	}

	public static int[] argmin(int[] x) {
		int n = x.length;
		if (n <= 0)
			return null;

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> Integer.compare(x[i], x[j])).mapToInt(e -> e)
				.toArray();

		int min = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == min).toArray();
	}

	public static <R extends Comparable<? super R>> int[] argmin(int[] x, Function<Integer, R> function) {
		int n = x.length;
		if (n <= 0)
			return null;

		List<R> result = new ArrayList<>(n);
		for (int elem : x)
			result.add(function.apply(elem));

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> result.get(i).compareTo(result.get(j)))
				.mapToInt(e -> e).toArray();

		R min = result.get(rank[0]);
		return Arrays.stream(rank).filter(i -> result.get(i) == min).toArray();
	}

	public static int[] argmin(float[] x) {
		int n = x.length;
		if (n <= 0)
			return null;

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> Float.compare(x[i], x[j])).mapToInt(e -> e)
				.toArray();

		float min = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == min).toArray();
	}

	public static <R extends Comparable<? super R>> int[] argmin(float[] x, Function<Float, R> function) {
		int n = x.length;
		if (n <= 0)
			return null;

		List<R> result = new ArrayList<>(n);
		for (float elem : x)
			result.add(function.apply(elem));

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> result.get(i).compareTo(result.get(j)))
				.mapToInt(e -> e).toArray();

		R min = result.get(rank[0]);
		return Arrays.stream(rank).filter(i -> result.get(i) == min).toArray();
	}

	public static int[] argmin(double[] x) {
		int n = x.length;
		if (n <= 0)
			return null;

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> Double.compare(x[i], x[j])).mapToInt(e -> e)
				.toArray();

		double min = x[rank[0]];
		return Arrays.stream(rank).filter(i -> x[i] == min).toArray();
	}

	public static <R extends Comparable<? super R>> int[] argmin(double[] x, Function<Double, R> function) {
		int n = x.length;
		if (n <= 0)
			return null;

		List<R> result = new ArrayList<>(n);
		for (double elem : x)
			result.add(function.apply(elem));

		int[] rank = IntStream.range(0, n).boxed().sorted((i, j) -> result.get(i).compareTo(result.get(j)))
				.mapToInt(e -> e).toArray();

		R min = result.get(rank[0]);
		return Arrays.stream(rank).filter(i -> result.get(i) == min).toArray();
	}

	/**
	 * @param list1
	 * @param list2
	 * @return intersection of two sets
	 */
	public static <T> List<T> intersect(Collection<T> set1, Collection<T> set2) {
		if (set1 == null || set2 == null)
			return null;
		List<T> intersect = new ArrayList<>();
		for (T id1 : set1) {
			if (set2.contains(id1))
				intersect.add(id1);
		}
		return intersect;
	}

	/**
	 * Sort List based on Reference
	 * 
	 * @param reference
	 * @param list
	 * @param ascend
	 */
	public static void linkedSort(List<? extends Number> reference, List<? extends Number> list, boolean ascend) {
		int len = reference.size();
		if (len <= 1)
			return;
		int inv = ascend ? 1 : -1;

		list.sort((a, b) -> inv * (Double.compare(reference.get(list.indexOf(a)).doubleValue(),
				reference.get(list.indexOf(b)).doubleValue())));
	}

	public static class Series {
		public static float[] linspace(float a, float b, int n) {
			float[] space = new float[n + 1];
			space[0] = a;
			float step = (b - a) / n;
			for (int i = 1; i <= n; i++)
				space[i] = space[i - 1] + step;
			return space;
		}

		public static double[] linspace(double a, double b, int n) {
			double[] space = new double[n];
			space[0] = a;
			double step = (b - a) / (n - 1);
			for (int i = 1; i < n; i++)
				space[i] = space[i - 1] + step;
			return space;
		}

		public static double[] range(double a, double step, int n) {
			double[] ret = new double[n];
			ret[0] = a;
			for (int i = 1; i < n; i++)
				ret[i] = ret[i - 1] + step;
			return ret;
		}

		public static float[] range(float a, float step, int n) {
			float[] ret = new float[n];
			ret[0] = a;
			for (int i = 1; i < n; i++)
				ret[i] = ret[i - 1] + step;
			return ret;
		}

		public static int[] range(int a, int step, int n) {
			int[] ret = new int[n];
			ret[0] = a;
			for (int i = 1; i < n; i++)
				ret[i] = ret[i - 1] + step;
			return ret;
		}

		/**
		 * @param a
		 * @param b
		 * @return
		 */
		public static int[] range(int a, int b) {
			if (b <= a)
				throw new IllegalArgumentException(String.format("%d <= %d", b, a));
			return range(a, 1, b - a);
		}
	}

	/**
	 * Constrain Within
	 * 
	 * @param val
	 * @param l
	 * @param u
	 * @return val constraint in [l, u]
	 */
	public static int constrain(int val, int l, int u) {
		if (val < l)
			val = l;
		if (val > u)
			val = u;
		return val;
	}

	public static long constrain(long val, long l, long u) {
		if (val < l)
			val = l;
		if (val > u)
			val = u;
		return val;
	}

	public static float constrain(float val, float l, float u) {
		if (val < l)
			val = l;
		if (val > u)
			val = u;
		return val;
	}

	public static double constrain(double val, double l, double u) {
		if (val < l)
			val = l;
		if (val > u)
			val = u;
		return val;
	}

	/**
	 * @param array
	 * @return list.get(i) = array[i]
	 */
	public static List<Double> toList(double[] array) {
		List<Double> list = new ArrayList<>();
		int len = array.length;
		for (int i = 0; i < len; i++)
			list.add(array[i]);
		return list;
	}

	public static List<Float> toList(float[] array) {
		List<Float> list = new ArrayList<>();
		int len = array.length;
		for (int i = 0; i < len; i++)
			list.add(array[i]);
		return list;
	}

	public static List<Integer> toList(int[] array) {
		List<Integer> list = new ArrayList<>();
		int len = array.length;
		for (int i = 0; i < len; i++)
			list.add(array[i]);
		return list;
	}

	public static <T> List<T> toList(T[] array) {
		return Arrays.asList(array);
	}
}