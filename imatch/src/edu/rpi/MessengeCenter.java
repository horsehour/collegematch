package edu.rpi;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeoutException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rabbitmq.client.AMQP;
import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
import com.rabbitmq.client.Consumer;
import com.rabbitmq.client.DefaultConsumer;
import com.rabbitmq.client.Envelope;

import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

public class MessengeCenter {
	private static final Logger logger = LoggerFactory.getLogger(MessengeCenter.class);
	private final static String QUEUE_NAME = "predictor_queue";

	JedisPool pool = null;
	CollegeMatch cm = null;

	public final int portRedis = 6379;

	public MessengeCenter() {
	}

	void train(String request, Jedis redis) {
		String[] args = request.split(" ");
		if (args.length == 5) {
			double[][] colleges = DataUtil.getRedisData(redis, args[1], true);
			double[][] students = DataUtil.getRedisData(redis, args[2], true);
			List<int[]> preferences = new ArrayList<>();
			DataUtil.getRedisData(redis, args[3], preferences);
			double[][] w = cm.learn(colleges, students, preferences);
			DataUtil.setRedisData(redis, args[4], w, false);
		} else
			logger.info("USAGE: -train <colleges> <students> <preferences> <model>");
	}

	void predict(String request, Jedis redis) {
		String[] args = request.split(" ");
		if (args.length == 6) {
			double[][] w = DataUtil.getRedisData(redis, args[1], false);
			double[][] colleges = DataUtil.getRedisData(redis, args[2], false);
			double[] student = new double[w.length];
			DataUtil.getRedisData(redis, args[3], false, student);
			int k = Integer.parseInt(args[4]);
			int[] preference = null;
			if (k <= 0)
				preference = cm.predict(student, colleges, w);
			else
				preference = cm.predict(student, colleges, w, k);
			DataUtil.setRedisData(redis, args[5], preference);
		} else
			logger.info("USAGE: -predict <model> <colleges> <student> <k> <preference>");
	}

	void enhance(String request, Jedis redis) {
		String[] args = request.split(" ");
		if (args.length == 6) {
			double[][] w = DataUtil.getRedisData(redis, args[1], false);
			double[][] colleges = DataUtil.getRedisData(redis, args[2], true);
			double[][] students = DataUtil.getRedisData(redis, args[3], true);
			List<int[]> preferences = new ArrayList<>();
			DataUtil.getRedisData(redis, args[4], preferences);
			w = cm.enhance(w, colleges, students, preferences);
			DataUtil.setRedisData(redis, args[5], w, false);
		} else
			logger.info("USAGE: -enhance <model> <colleges> <students> <preferences> <enhance>");
	}

	void simulate(String request, Jedis redis) {
		String[] args = request.split(" ");
		if (args.length == 4) {
			int ns = 100, ds = 25;
			int nc = 200, dc = 23;
			int maxK = 10;

			double[][] students = DataUtil.generateFeatureMatrix(ns, ds, -1, 1);
			double[][] colleges = DataUtil.generateFeatureMatrix(nc, dc, 0, 1);

			List<int[]> preferences = DataUtil.generatePreferences(students, colleges, maxK);

			double[][] w = cm.learn(colleges, students, preferences);
			DataUtil.setRedisData(redis, args[0], colleges, true);
			DataUtil.setRedisData(redis, args[1], students, true);
			DataUtil.setRedisData(redis, args[2], preferences);
			DataUtil.setRedisData(redis, args[3], w, true);
		} else
			logger.info("USAGE: -simulate <colleges> <students> <preferences> <model>");
	}

	class QueryConsumer extends DefaultConsumer {
		private final Logger loggerQuery = LoggerFactory.getLogger(MessengeCenter.class);

		public QueryConsumer(Channel channel) {
			super(channel);
		}

		@Override
		public void handleDelivery(String consumerTag, Envelope envelope, AMQP.BasicProperties properties, byte[] body)
				throws IOException {
			String request = new String(body, "UTF8");

			try (Jedis redis = pool.getResource()) {
				if (request.startsWith("-train"))
					train(request, redis);
				else if (request.startsWith("-predict"))
					predict(request, redis);
				else if (request.startsWith("-enhance"))
					enhance(request, redis);
				else if (request.startsWith("-simulate"))
					simulate(request, redis);
			}
			loggerQuery.info("Received " + request);
		}
	}

	/**
	 * Started up RabbitMQ listener
	 */
	public void run(String hostRedis, String hostRabbitMQ) {
		if (hostRedis == null || hostRedis.isEmpty())
			hostRedis = "localhost";
		if (hostRabbitMQ == null || hostRabbitMQ.isEmpty())
			hostRabbitMQ = "localhost";

		cm = new CollegeMatch();
		pool = new JedisPool(new JedisPoolConfig(), hostRedis, portRedis);

		ConnectionFactory factory = new ConnectionFactory();
		factory.setHost(hostRabbitMQ);

		Connection connection;
		try {
			connection = factory.newConnection();
			Channel channel = connection.createChannel();

			boolean durable = false;
			channel.queueDeclare(QUEUE_NAME, durable, false, false, null);

			logger.info("Waiting for messages from " + QUEUE_NAME);
			Consumer consumer = new QueryConsumer(channel);

			boolean autoAck = true;
			// loop that waits for message
			channel.basicConsume(QUEUE_NAME, autoAck, consumer);
		} catch (IOException e) {
			logger.info("RabbitMQ server is down.");
			logger.info(e.getMessage());
		} catch (TimeoutException e) {
			logger.info(e.getMessage());
		}
	}

	public static void main(String[] args) {
		MessengeCenter mc = new MessengeCenter();
		if (args == null || !args[0].equals("-run")) {
			System.err.println("USAGE: -run <redis_host> <rabbitmq_host>");
			return;
		}

		int len = args.length;
		if (len == 1)
			mc.run(null, null);
		if (len == 2)
			mc.run(args[1], null);
		else if (len == 3)
			mc.run(args[1], args[2]);
	}
}