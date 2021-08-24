package minsait.ttaa.datio.engine;

import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import org.jetbrains.annotations.NotNull;

import static minsait.ttaa.datio.common.Common.*;
import static minsait.ttaa.datio.common.naming.PlayerInput.*;
import static minsait.ttaa.datio.common.naming.PlayerOutput.*;
import static org.apache.spark.sql.functions.*;

public class Transformer extends Writer {
    private SparkSession spark;

    public Transformer(@NotNull SparkSession spark) {
        this.spark = spark;
        Dataset<Row> df = readInput();

        df.printSchema();

        df = cleanData(df);
        df = rankoverWindowFunction(df);
        df = potentialDivideByOverall(df);
        df = filterPlayerCatAndPotentialVsOverall(df);
        df = columnSelection(df);

        // for show 100 records after your transformations and show the Dataset schema
        df.show(100, false);
        df.printSchema();

        // Uncomment when you want write your final output
        //write(df);
    }

    private Dataset<Row> columnSelection(Dataset<Row> df) {
        return df.select(
                shortName.column(),
                longName.column(),
                age.column(),
                heightCm.column(),
                weightKg.column(),
                nationality.column(),
                clubName.column(),
                overall.column(),
                potential.column(),
                teamPosition.column(),
                playerCat.column(),
                potentialVsOverall.column()
        );
    }

    /**
     * @return a Dataset readed from csv file
     */
    private Dataset<Row> readInput() {
        Dataset<Row> df = spark.read()
                .option(HEADER, true)
                .option(INFER_SCHEMA, true)
                .csv(INPUT_PATH);
        return df;
    }

    /**
     * @param df
     * @return a Dataset with filter transformation applied
     * column team_position != null && column short_name != null && column overall != null
     */
    private Dataset<Row> cleanData(Dataset<Row> df) {
        df = df.filter(
                teamPosition.column().isNotNull().and(
                        shortName.column().isNotNull()
                ).and(
                        overall.column().isNotNull()
                )
        );

        return df;
    }

    /**
     * @param df is a Dataset with players information (must have nationality, team_position and overall columns)
     * @return add to the Dataset the column "player_cat"
     * by each position value
     * cat A for if is in 3 players
     * cat B for if is in 5 players
     * cat C for if is in 10 players
     * cat D for the rest
     */
    private Dataset<Row> rankoverWindowFunction(Dataset<Row> df) {
        WindowSpec w = Window
                .partitionBy(nationality.column(), teamPosition.column())
                .orderBy(overall.column());

        Column rank = rank().over(w);

        Column rule = when(rank.$less(3), "A")
                .when(rank.$less(5), "B")
                .when(rank.$less(10), "C")
                .otherwise("D");

        df = df.withColumn(playerCat.getName(), rule);

        return df;
    }


    /**
     * @param df is a Dataset with players information (must have potential and overall columns)
     * @return add to the Dataset the column "potential_vs_overall"
     * by each overall value
     * column potential divide by column overall
     */
    private Dataset<Row> potentialDivideByOverall(Dataset<Row> df) {
        df = df.withColumn(potentialVsOverall.getName(), potential.column().divide(overall.column()));

        return df;
    }

    /**
     * @param df
     * @return a Dataset with filter transformation applied
     * column player_cat in A , B
     * or if column player_cat = C && potential_vs_overall > 1.15
     * or if column player_cat = D && column potential_vs_overall > 1.25
     */
    private Dataset<Row> filterPlayerCatAndPotentialVsOverall(Dataset<Row> df) {

        df = df.filter(playerCat.column().startsWith("A").or(playerCat.column().startsWith("B")).or(
                        playerCat.column().startsWith("C").and(potentialVsOverall.column().gt(1.15))
                ).or(
                        playerCat.column().startsWith("D").and(potentialVsOverall.column().gt(1.25))
                )
        );

        return df;
    }


}
