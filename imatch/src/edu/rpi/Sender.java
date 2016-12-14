package edu.rpi;

import com.rabbitmq.client.ConnectionFactory;
import com.rabbitmq.client.Connection;
import java.io.IOException;
import java.util.concurrent.TimeoutException;
import com.rabbitmq.client.Channel;

public class Sender {
	private final static String QUEUE_NAME = "predictor_queue";

	public static void main(String[] argv) {
		ConnectionFactory factory = new ConnectionFactory();
		factory.setHost("localhost");
		
		Connection connection;
		try {
			connection = factory.newConnection();

			Channel channel = connection.createChannel();
			channel.queueDeclare(QUEUE_NAME, false, false, false, null);

			String message = "-predict";
			channel.basicPublish("", QUEUE_NAME, null, message.getBytes());
			System.out.println("Java Queue - Message RabbitMQ Java Sent: '" + message + "'");

			channel.close();
			connection.close();
		} catch (IOException e) {
			System.out.println("RabbitMQ server is Down !");
			System.out.println(e.getMessage());
		} catch (TimeoutException e) {
			e.printStackTrace();
		}
	}
}