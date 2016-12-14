package edu.rpi;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;

import redis.clients.jedis.Jedis;

/**
 * @author Chunheng Jiang
 * @version 1.0
 * @since 5:41:03 PM, Nov 17, 2016
 */

public class DataUtil {
	public static OpenOption[] options = { StandardOpenOption.CREATE, StandardOpenOption.WRITE };

	/**
	 * Generate feature matrix based on uniform distribution
	 * 
	 * @param n
	 * @param m
	 * @param a
	 * @param b
	 * @return Randomly generated feature matrix
	 */
	public static double[][] generateFeatureMatrix(int n, int m, double a, double b) {
		double[][] features = new double[n][m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++)
				features[i][j] = MathLib.Rand.uniform(a, b);
		}
		return features;
	}

	/**
	 * Generate preference rankings
	 * 
	 * @param students
	 * @param colleges
	 * @param maxK
	 *            maximum length of each preference ranking
	 * @return randomly generated preference rankings
	 */
	public static List<int[]> generatePreferences(double[][] students, double[][] colleges, int maxK) {
		int ds = students[0].length, dc = colleges[0].length;
		int d = 10;
		double[][] ws = new double[d][ds];
		double[][] wc = new double[d][dc];
		for (int c = 0; c < d; c++) {
			ws[c] = MathLib.Rand.distribution(ds);
			wc[c] = MathLib.Rand.distribution(dc);
		}

		List<int[]> preferences = new ArrayList<>();
		int ns = students.length, nc = colleges.length;
		double[] sim = new double[nc];

		/**
		 * 3 <= k <= maxK
		 */
		List<Integer> ks = MathLib.Rand.sampleR(3, maxK + 1, ns);

		for (int i = 0; i < ns; i++) {
			double[] xs = students[i];
			for (int j = 0; j < nc; j++) {
				double[] xc = colleges[j];
				sim[j] = MathLib.Matrix.dotProd(MathLib.Matrix.ax(ws, xs), MathLib.Matrix.ax(wc, xc));
			}
			int[] rank = MathLib.getRank(sim, false);
			int k = ks.get(i);
			preferences.add(Arrays.copyOf(rank, k));
		}
		return preferences;
	}

	/**
	 * Load the training data from comma-separated-values (csv) source, where
	 * the first column is the index of the corresponding row entry.
	 * 
	 * @param src
	 * @return training data matrix
	 */
	public static double[][] loadTrainingData(String src) {
		List<String> lines = null;
		try {
			lines = Files.readAllLines(Paths.get(src));
		} catch (IOException e) {
			System.err.println(e.getMessage());
			String msg = String.format("ERROR: Failed to read file \"%s\".", src);
			System.err.println(msg);
			return null;
		}

		int n = lines.size();
		double[][] data = new double[n][];
		int d = -1;
		for (int i = 0; i < n; i++) {
			String line = lines.get(i);
			String[] fields = line.split(",");

			if (d == -1)
				d = fields.length - 1;

			data[i] = new double[d];
			for (int k = 1; k <= d; k++)
				data[i][k - 1] = Double.parseDouble(fields[k]);
		}
		return data;
	}

	/**
	 * Load input instance for prediction
	 * 
	 * @param src
	 * @return input instance
	 */
	public static double[] loadInstance(String src) {
		List<String> lines = null;
		try {
			lines = Files.readAllLines(Paths.get(src));
		} catch (IOException e) {
			System.err.println(e.getMessage());
			String msg = String.format("ERROR: Failed to read file \"%s\".", src);
			System.err.println(msg);
			return null;
		}

		String line = lines.get(0);
		String[] fields = line.split(",");

		int d = fields.length - 1;
		double[] instance = new double[d];
		for (int i = 1; i <= d; i++)
			instance[i - 1] = Double.parseDouble(fields[i]);
		return instance;
	}

	public static double[][] loadModel(String src) {
		List<String> lines = null;
		try {
			lines = Files.readAllLines(Paths.get(src));
		} catch (IOException e) {
			System.err.println(e.getMessage());
			String msg = String.format("ERROR: Failed to read file \"%s\".", src);
			System.err.println(msg);
			return null;
		}

		int n = lines.size();
		double[][] data = new double[n][];
		int d = -1;
		for (int i = 0; i < n; i++) {
			String line = lines.get(i);
			String[] fields = line.split(",");

			if (d == -1)
				d = fields.length;

			data[i] = new double[d];
			for (int k = 0; k < d; k++)
				data[i][k] = Double.parseDouble(fields[k]);
		}
		return data;
	}

	/**
	 * First column in each row contains the index or name of the row entry
	 * 
	 * @param preferenceFile
	 * @return ranking preference
	 */
	public static List<int[]> loadPreferences(String preferenceFile) {
		List<String> lines = null;
		try {
			lines = Files.readAllLines(Paths.get(preferenceFile));
		} catch (IOException e) {
			System.err.println(e.getMessage());
			String msg = String.format("ERROR: Failed to read file \"%s\".", preferenceFile);
			System.err.println(msg);
			return null;
		}

		List<int[]> preferences = new ArrayList<>();
		int n = lines.size();
		for (int i = 0; i < n; i++) {
			String line = lines.get(i);
			String[] fields = line.split(",");
			int len = fields.length - 1;
			int[] pref = new int[len];
			for (int k = 1; k <= len; k++)
				pref[k - 1] = Integer.parseInt(fields[k]);
			preferences.add(pref);
		}
		return preferences;
	}

	/**
	 * Store data to specific place
	 * 
	 * @param data
	 * @param dest
	 */
	public static void write(double[][] data, String dest) {
		int n = data.length, m = data[0].length;

		List<String> lines = new ArrayList<>();
		StringBuffer sb = null;
		for (int i = 0; i < n; i++) {
			sb = new StringBuffer();
			sb.append(i);
			for (int j = 0; j < m; j++)
				sb.append("," + data[i][j]);
			lines.add(sb.toString());
		}

		try {
			Files.write(Paths.get(dest), lines, options);
		} catch (IOException e) {
			System.err.println(e.getMessage());
			String msg = String.format("ERROR: Failed to write to file \"%s\".", dest);
			System.err.println(msg);
			return;
		}
	}

	/**
	 * Store data to specific place
	 * 
	 * @param data
	 * @param dest
	 */
	public static void write(double[] data, String dest) {
		List<String> lines = new ArrayList<>();
		StringBuffer sb = new StringBuffer();
		sb.append(data[0]);
		for (int i = 1; i < data.length; i++)
			sb.append("," + data[i]);
		lines.add("0," + sb.toString());

		try {
			Files.write(Paths.get(dest), lines, options);
		} catch (IOException e) {
			System.err.println(e.getMessage());
			String msg = String.format("ERROR: Failed to write to file \"%s\".", dest);
			System.err.println(msg);
			return;
		}
	}

	public static void writeModel(double[][] data, String dest) {
		int n = data.length, m = data[0].length;

		List<String> lines = new ArrayList<>();
		StringBuffer sb = null;
		for (int i = 0; i < n; i++) {
			sb = new StringBuffer();
			sb.append(data[i][0]);
			for (int j = 1; j < m; j++)
				sb.append("," + data[i][j]);
			lines.add(sb.toString());
		}

		try {
			Files.write(Paths.get(dest), lines, options);
		} catch (IOException e) {
			System.err.println(e.getMessage());
			String msg = String.format("ERROR: Failed to write to file \"%s\".", dest);
			System.err.println(msg);
			return;
		}
	}

	public static void writePreference(int[] pref, String dest) {
		List<String> lines = new ArrayList<>();
		StringBuffer sb = new StringBuffer();
		sb.append(pref[0]);
		for (int i = 1; i < pref.length; i++)
			sb.append("," + pref[i]);
		lines.add("0," + sb.toString());

		try {
			Files.write(Paths.get(dest), lines, options);
		} catch (IOException e) {
			System.err.println(e.getMessage());
			String msg = String.format("ERROR: Failed to write to file \"%s\".", dest);
			System.err.println(msg);
			return;
		}
	}

	public static void writePreferences(List<int[]> preferens, String dest) {
		int n = preferens.size();

		List<String> lines = new ArrayList<>();
		StringBuffer sb = null;
		int[] pref;
		for (int i = 0; i < n; i++) {
			pref = preferens.get(i);
			sb = new StringBuffer();
			sb.append(i);
			for (int j = 0; j < pref.length; j++)
				sb.append("," + pref[j]);
			lines.add(sb.toString());
		}

		try {
			Files.write(Paths.get(dest), lines, options);
		} catch (IOException e) {
			System.err.println(e.getMessage());
			String msg = String.format("ERROR: Failed to write to file \"%s\".", dest);
			System.err.println(msg);
			return;
		}
	}

	/**
	 * http://stackoverflow.com/questions/134492/how-to-serialize-an-object-
	 * into-a-string
	 * 
	 * Deserialize from string an object instance Serialize object and output as
	 * string
	 * 
	 * @param o
	 * @return Serialized object
	 * @throws IOException
	 */
	public static String serialize(Serializable o) throws IOException {
		ByteArrayOutputStream bos = new ByteArrayOutputStream();
		ObjectOutputStream oos = new ObjectOutputStream(bos);
		oos.writeObject(o);
		oos.close();
		return Base64.getEncoder().encodeToString(bos.toByteArray());
	}

	/**
	 * http://stackoverflow.com/questions/134492/how-to-serialize-an-object-
	 * into-a-string Deserialize from string an object instance
	 * 
	 * @param s
	 * @return Object instance
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static Object deserialize(String s) throws IOException, ClassNotFoundException {
		byte[] data = Base64.getDecoder().decode(s);
		ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(data));
		Object o = ois.readObject();
		ois.close();
		return o;
	}

	/**
	 * 
	 * @param redis
	 * @param key
	 * @param hasRowIndex
	 *            true - row are indexed (first column); false otherwise
	 * @return data in redis
	 */
	public static double[][] getRedisData(Jedis redis, String key, boolean hasRowIndex) {
		String val = redis.get(key);
		String[] lines = val.split("\n");
		int n = lines.length;
		double[][] data = new double[n][];
		int f = hasRowIndex ? 1 : 0;

		int d = -1;
		for (int i = 0; i < n; i++) {
			String line = lines[i];
			String[] fields = line.split(",");

			if (d == -1)
				d = fields.length - f;

			data[i] = new double[d];
			for (int k = f; k <= d + f - 1; k++)
				data[i][k - f] = Double.parseDouble(fields[k]);
		}
		return data;
	}

	public static void getRedisData(Jedis redis, String key, boolean hasRowIndex, double[] data) {
		String val = redis.get(key);
		String[] lines = val.split(",");

		int f = hasRowIndex ? 1 : 0;
		int n = lines.length - f;

		for (int i = f; i < n; i++)
			data[i - f] = Double.parseDouble(lines[i]);
	}

	/**
	 * data is serialized and kept in redis
	 * 
	 * @param redis
	 * @param key
	 * @param data
	 */
	public static void getRedisData(Jedis redis, String key, double[][] data) {
		String val = redis.get(key);
		try {
			data = (double[][]) deserialize(val);
		} catch (ClassNotFoundException | IOException e) {
			e.printStackTrace();
		}
	}

	public static void getRedisData(Jedis redis, String key, List<int[]> data) {
		String val = redis.get(key);
		String[] lines = val.split("\n");
		int n = lines.length;

		int d = -1;
		for (int i = 0; i < n; i++) {
			String line = lines[i];
			String[] fields = line.split(",");

			if (d == -1)
				d = fields.length - 1;

			int[] row = new int[d];
			for (int k = 1; k <= d; k++)
				row[k - 1] = Integer.parseInt(fields[k]);
			data.add(row);
		}
	}

	/**
	 * @param redis
	 * @param key
	 * @param data
	 * @param serialized
	 */
	public static void setRedisData(Jedis redis, String key, double[][] data, boolean serialized) {
		int n = data.length, m = data[0].length;

		if (!serialized) {
			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < n; i++) {
				sb.append(data[i][0]);
				for (int j = 1; j < m; j++)
					sb.append("," + data[i][j]);
				sb.append("\n");
			}
			redis.set(key, sb.toString());
			return;
		}

		try {
			String val = serialize(data);
			redis.set(key, val);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void setRedisData(Jedis redis, String key, List<double[]> data, boolean string) {
		int n = data.size(), m = data.get(0).length;

		if (string) {
			StringBuffer sb = new StringBuffer();
			for (int i = 0; i < n; i++) {
				sb.append(data.get(i)[0]);
				for (int j = 1; j < m; j++)
					sb.append("," + data.get(i)[j]);
				sb.append("\n");
			}
			redis.set(key, sb.toString());
			return;
		}
		try {
			String val = serialize((Serializable) data);
			redis.set(key, val);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void setRedisData(Jedis redis, String key, List<int[]> data) {
		try {
			String val = serialize((Serializable) data);
			redis.set(key, val);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void setRedisData(Jedis redis, String key, double[] data, boolean string) {
		int n = data.length;

		if (string) {
			StringBuffer sb = new StringBuffer();
			sb.append(data[0]);
			for (int i = 1; i < n; i++)
				sb.append("," + data[i]);
			redis.set(key, sb.toString());
			return;
		}

		try {
			String val = serialize(data);
			redis.set(key, val);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void setRedisData(Jedis redis, String key, int[] data) {
		int n = data.length;

		StringBuffer sb = new StringBuffer();
		sb.append(data[0]);
		for (int i = 1; i < n; i++)
			sb.append("," + data[i]);
		redis.set(key, sb.toString());
	}
}